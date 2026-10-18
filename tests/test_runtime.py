"""The runtime against the official package on the CPU, with the real model (34 MB)."""

import numpy as np
import pytest

from ema_comfy import runtime

TEXT = "15 Ekim 2026 Çarşamba günü saat 14:30'da toplantı başlayacak. Lütfen zamanında gelin."


@pytest.fixture(scope="module")
def tts(model_dir):
    return runtime.acquire(runtime.make_handle(str(model_dir), "cpu", "4", False))


def _official(model_dir, text, seed, speed=1.0, rate=48000, batch=4):
    """The package's own objects built from the same files, as EMA() builds them after downloading."""
    from ema_lightning import EMA
    from ema_lightning.decoder import load_decoder
    from ema_lightning.frontend import Frontend
    from ema_lightning.model import load_acoustic

    model = load_acoustic(str(model_dir / "ema.pt"), "cpu")
    official = EMA._from_parts(model, load_decoder(str(model_dir / "decoder.pt"), "cpu"), Frontend(model.vocab), "cpu")
    official._batch_size = batch
    return official.say(text, speed=speed, seed=seed, sample_rate=rate)


def test_options_match_the_package():
    from ema_lightning.audio import RATES

    assert runtime.SAMPLE_RATES == RATES and runtime.BATCH_SIZES[0] == "auto"


def test_handle_validation(model_dir):
    with pytest.raises(ValueError):
        runtime.make_handle(str(model_dir), "cpu", "3", False)
    with pytest.raises(ValueError):
        runtime.make_handle(str(model_dir), "cpu", "auto", True)  # lightning needs CUDA


def test_say_equals_the_package(tts, model_dir):
    out = runtime.say(tts, TEXT, seed=5)
    ref = _official(model_dir, TEXT, 5)
    assert out["sample_rate"] == 48000 and out["audio"].dtype == np.float32 and np.array_equal(out["audio"], ref.audio)
    assert out["words"] == [{"text": w.text, "start": w.start, "end": w.end} for w in ref.words]
    assert out["report"]["duration"] == round(ref.duration, 3) and out["report"]["batch_size"] == 4


def test_numbers_are_read_out(tts):
    words = [w["text"] for w in runtime.say(tts, "Saat 14:30", seed=0)["words"]]
    assert words and not any(ch.isdigit() for w in words for ch in w)  # the normalizer turned the numbers into words
    starts = [w["start"] for w in runtime.say(tts, TEXT, seed=0)["words"]]
    assert starts == sorted(starts)


@pytest.mark.parametrize("rate", [24000, 16000, 8000])
def test_sample_rates(tts, model_dir, rate):
    out = runtime.say(tts, "Merhaba dünya.", seed=1, sample_rate=rate)
    assert out["sample_rate"] == rate and np.array_equal(out["audio"], _official(model_dir, "Merhaba dünya.", 1, rate=rate).audio)


def test_seed_speed_and_repeat(tts):
    a = runtime.say(tts, "Merhaba dünya.", seed=1)["audio"]
    assert np.array_equal(a, runtime.say(tts, "Merhaba dünya.", seed=1)["audio"])
    assert not np.array_equal(a, runtime.say(tts, "Merhaba dünya.", seed=2)["audio"])
    fast = runtime.say(tts, TEXT, seed=1, speed=2.0)["report"]["duration"]
    normal = runtime.say(tts, TEXT, seed=1)["report"]["duration"]
    assert fast < normal


def test_bad_input(tts):
    with pytest.raises(ValueError):
        runtime.say(tts, "   ", seed=0)
    with pytest.raises(ValueError):
        runtime.say(tts, "Merhaba", seed=0, sample_rate=44100)
    with pytest.raises(ValueError):
        runtime.say(tts, "Merhaba", seed=0, speed=9.0)


def test_models_are_kept_per_device_and_batch_size_is_per_call(model_dir):
    a = runtime.acquire(runtime.make_handle(str(model_dir), "cpu", "2", False))
    b = runtime.acquire(runtime.make_handle(str(model_dir), "cpu", "8", False))
    assert a is b and b._batch_size == 8
