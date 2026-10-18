"""The node pack as ComfyUI loads it: registration, model discovery, ComfyUI's own validation of the shipped workflows, and the nodes
on the CPU with the real model."""

import asyncio
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"EMALightningLoader", "EMALightningSay"}


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, model_dir, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    folder_paths.set_output_directory(str(base / "output"))
    models = base / "models" / "ema_lightning"
    shutil.copytree(model_dir, models / "ema-lightning", ignore=shutil.ignore_patterns("assets", ".cache", "*.md"))  # the name the workflows use
    (models / "incomplete").mkdir()
    (models / "incomplete" / "ema.pt").write_bytes(b"")
    folder_paths.add_model_folder_path("ema_lightning", str(models), is_default=True)
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    return nodes


def pack_module(comfy_nodes, name):
    return importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS["EMALightningSay"].__module__.rsplit(".", 1)[0] + "." + name)


def say(comfy_nodes, text="Merhaba dünya.", seed=0, speed=1.0, rate="48000", batch="4"):
    m = comfy_nodes.NODE_CLASS_MAPPINGS
    handle = m["EMALightningLoader"].execute(model="ema-lightning", device="cpu", batch_size=batch, lightning=False).args[0]
    return m["EMALightningSay"].execute(model=handle, text=text, seed=seed, speed=speed, sample_rate=rate).args


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_only_complete_model_folders_are_listed(comfy_nodes):
    paths = pack_module(comfy_nodes, "paths")
    assert paths.models() == ["ema-lightning"]
    for bad in ("incomplete", "../ema-lightning", "missing"):
        with pytest.raises(FileNotFoundError):
            paths.model_path(bad)


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_say_node_outputs_comfy_audio(comfy_nodes, model_dir):
    audio, words, report = say(comfy_nodes, "Saat 14:30'da buluşalım.", seed=3)
    assert audio["sample_rate"] == 48000 and audio["waveform"].shape[:2] == (1, 1) and audio["waveform"].dtype == torch.float32
    parsed, timing = json.loads(report), json.loads(words)
    assert parsed["seed"] == 3 and parsed["batch_size"] == 4 and timing and all(w["end"] > w["start"] for w in timing)
    assert abs(audio["waveform"].shape[-1] / 48000 - parsed["duration"]) < 1e-3
    runtime = pack_module(comfy_nodes, "runtime")
    direct = runtime.say(runtime.acquire(runtime.make_handle(str(model_dir), "cpu", "4", False)), "Saat 14:30'da buluşalım.", 3)
    assert np.array_equal(audio["waveform"].reshape(-1).numpy(), direct["audio"])


def test_sample_rate_option(comfy_nodes):
    audio, _, _ = say(comfy_nodes, rate="16000")
    assert audio["sample_rate"] == 16000


def test_save_audio_node_writes_the_file(comfy_nodes, monkeypatch):
    """ComfyUI's Save Audio (Advanced) accepts the node's AUDIO (called directly here, so without the prompt metadata it would embed)."""
    import comfy.cli_args
    import folder_paths

    monkeypatch.setattr(comfy.cli_args.args, "disable_metadata", True)
    audio, _, _ = say(comfy_nodes, seed=4)
    save = comfy_nodes.NODE_CLASS_MAPPINGS["SaveAudioAdvanced"]
    save.execute(audio=audio, filename_prefix="ema_test/clip", format={"format": "flac"})
    files = list((Path(folder_paths.get_output_directory()) / "ema_test").glob("clip*.flac"))
    assert len(files) == 1 and files[0].stat().st_size > 1000


def test_empty_text_and_interrupt(comfy_nodes, monkeypatch):
    import comfy.model_management as mm

    with pytest.raises(ValueError):
        say(comfy_nodes, "  ")

    def interrupt():
        raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        say(comfy_nodes)


def test_shipped_gui_workflows_use_known_nodes():
    for path in sorted((ROOT / "workflows" / "gui").glob("*.json")):
        types = {n["type"] for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}
        assert types <= NODE_IDS | {"SaveAudioAdvanced", "PreviewAny", "PreviewAudio"}, path.name
