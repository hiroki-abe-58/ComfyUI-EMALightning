"""ComfyUI nodes for EMA Lightning, a small and fast Turkish text-to-speech model (runs in the ComfyUI process; nothing is downloaded
at run time)."""

from __future__ import annotations

import json

import torch
from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

EMAModel = io.Custom("EMA_LIGHTNING_MODEL")


class EMALightningLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EMALightningLoader",
            display_name="EMA Lightning Loader",
            category="loaders/ema_lightning",
            description=("Loads EMA Lightning (Turkish text to speech: an 8.6M-parameter flow-matching model and vocoder, 48 kHz) from "
                         "models/ema_lightning through its official package. The model stays loaded for the session (34 MB)."),
            inputs=[
                io.Combo.Input("model", options=paths.models(), tooltip="A folder in models/ema_lightning with ema.pt and decoder.pt"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="The CPU also works, about 6x faster than real time."),
                io.Combo.Input("batch_size", options=list(runtime.BATCH_SIZES), default="auto",
                               tooltip="How many sentences or audio windows the package's scheduler runs at once. auto = the package's "
                                       "default (measured once per device and cached by the package; the measured value can vary between "
                                       "installs). The audio can differ very slightly (about 1e-3) between batch sizes when a text needs "
                                       "more audio windows than the batch holds; pick a fixed size for audio that repeats everywhere."),
                io.Boolean.Input("lightning", default=False,
                                 tooltip="The package's fast CUDA-graph mode, recorded once per session (minutes the first time). Not "
                                         "bit-identical to the plain path (the package accepts up to 1e-2), and it switches on cuDNN "
                                         "benchmark mode for the whole ComfyUI process."),
            ],
            outputs=[EMAModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, device, batch_size, lightning) -> io.NodeOutput:
        return io.NodeOutput(runtime.make_handle(paths.model_path(model), device, batch_size, lightning))


class EMALightningSay(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EMALightningSay",
            display_name="EMA Lightning Text to Speech",
            category="audio/ema_lightning",
            description=("Turkish speech from text. Numbers, dates, amounts and symbols are read out by the package's Turkish normalizer; "
                         "long texts are split into sentences. Outputs the audio and the time of every spoken word."),
            inputs=[
                EMAModel.Input("model"),
                io.String.Input("text", multiline=True, default="Merhaba, size nasıl yardımcı olabilirim?"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=True,
                             tooltip="Same text, seed, settings and device give the same audio."),
                io.Float.Input("speed", default=1.0, min=0.25, max=4.0, step=0.05, tooltip="Speaking rate (1 = normal)."),
                io.Combo.Input("sample_rate", options=[str(r) for r in runtime.SAMPLE_RATES], default="48000",
                               tooltip="48000 is the model's rate; the others are resampled by the package."),
            ],
            outputs=[io.Audio.Output(display_name="audio"), io.String.Output(display_name="words"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, text, seed, speed, sample_rate) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        out = runtime.say(runtime.acquire(model), text, seed, speed, int(sample_rate))
        audio = {"waveform": torch.from_numpy(out["audio"]).reshape(1, 1, -1), "sample_rate": out["sample_rate"]}
        return io.NodeOutput(audio, json.dumps(out["words"], ensure_ascii=False), json.dumps(out["report"], indent=1))


class EMALightningExtension(ComfyExtension):
    async def get_node_list(self):
        return [EMALightningLoader, EMALightningSay]


async def comfy_entrypoint() -> EMALightningExtension:
    paths.register()
    return EMALightningExtension()
