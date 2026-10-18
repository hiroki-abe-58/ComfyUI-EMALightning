"""Test setup. References come from environment variables, never from this repository:

EMA_COMFYUI_DIR          ComfyUI checkout (v0.38.0 or later)
EMA_MODEL_DIR            folder with the official ema.pt and decoder.pt (tools/download_model.py; 34 MB, so the tests use the real model)
EMA_REQUIRE_REFERENCES=1 fail instead of skip when one of them is missing (CI sets this)
The tests run on the CPU.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("EMA_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("EMA_COMFYUI_DIR")
MODEL = _path("EMA_MODEL_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (EMA_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (EMA_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def model_dir():
    return Path(_need(MODEL, "EMA Lightning model folder (EMA_MODEL_DIR)"))
