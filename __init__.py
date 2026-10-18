"""EMA Lightning for ComfyUI (unofficial integration of https://huggingface.co/canberkkkkkk/ema-lightning)."""

import logging

try:
    from .ema_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[EMA Lightning] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
