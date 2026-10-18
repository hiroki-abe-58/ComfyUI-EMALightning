"""EMA Lightning (Turkish text to speech) through its official package, ema-lightning 1.1.2.

The speech is made by the package's own `EMA.say()`: its Turkish text normalizer, sentence chunker, model, scheduler and decoder.
This module only builds the `EMA` object from local files (the package's documented constructor downloads from the Hub; here the
package's own loaders and `EMA._from_parts` are used instead) and passes the documented options through. Nothing here downloads
anything.

The package runs a background scheduler thread per `EMA` object that keeps its model alive, so a loaded model is kept for the
whole session (one per model folder, device and lightning setting); the model is 34 MB.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

BATCH_SIZES = ("auto", "1", "2", "4", "8", "16", "32", "64", "128")
SAMPLE_RATES = (48000, 24000, 16000, 8000)  # ema_lightning.audio.RATES

_LOCK = threading.Lock()
_MODELS: dict[tuple, object] = {}


def make_handle(folder: str, device: str, batch_size: str, lightning: bool) -> dict:
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {BATCH_SIZES}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device cuda was chosen but CUDA is not available")
    if lightning and device != "cuda":
        raise ValueError("lightning (compiled CUDA graphs) needs device cuda")
    return {"folder": folder, "device": device, "batch_size": batch_size, "lightning": bool(lightning)}


def acquire(handle: dict):
    """The package's EMA object for this model folder, device and lightning setting (built once per session)."""
    key = (handle["folder"], handle["device"], handle["lightning"])
    with _LOCK:
        tts = _MODELS.get(key)
        if tts is None:
            # ComfyUI runs nodes under torch.inference_mode(); build the weights as ordinary tensors
            with torch.inference_mode(False):
                tts = _load(*key)
            _MODELS[key] = tts
    # "auto" is the package's own choice: measured once per device and cached by the package (EMA.best_batch_size)
    tts._batch_size = None if handle["batch_size"] == "auto" else int(handle["batch_size"])
    return tts


def _load(folder: str, device: str, lightning: bool):
    from ema_lightning import EMA
    from ema_lightning.decoder import load_decoder
    from ema_lightning.frontend import Frontend
    from ema_lightning.model import load_acoustic

    model = load_acoustic(str(Path(folder) / "ema.pt"), device)
    tts = EMA._from_parts(model, load_decoder(str(Path(folder) / "decoder.pt"), device), Frontend(model.vocab), device)
    if lightning:
        tts.lightning()
    return tts


def say(tts, text: str, seed: int, speed: float = 1.0, sample_rate: int = 48000) -> dict:
    """{"audio": float32 [N] in -1..1, "sample_rate", "words": [{"text", "start", "end"}], "report": dict}."""
    if not text.strip():
        raise ValueError("text is empty")
    if sample_rate not in SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}")
    started = time.perf_counter()
    speech = tts.say(text, speed=float(speed), seed=int(seed), sample_rate=int(sample_rate))
    seconds = time.perf_counter() - started
    words = [{"text": w.text, "start": w.start, "end": w.end} for w in speech.words]
    report = {"seed": speech.seed, "speed": float(speed), "sample_rate": speech.sample_rate, "duration": round(speech.duration, 3),
              "words": len(words), "batch_size": tts._batch_size, "device": str(tts.device), "lightning": tts._engine.graphs is not None,
              "seconds": round(seconds, 3)}
    if seconds > 0 and speech.duration > 0:
        report["real_time_factor"] = round(seconds / speech.duration, 4)
    return {"audio": speech.audio, "sample_rate": speech.sample_rate, "words": words, "report": report}
