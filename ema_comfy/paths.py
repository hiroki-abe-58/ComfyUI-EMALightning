"""Model locations: `<ComfyUI models>/ema_lightning/<name>/` (plus any path added for `ema_lightning` in extra_model_paths.yaml), each
holding the official `ema.pt` (acoustic model) and `decoder.pt` (vocoder) of canberkkkkkk/ema-lightning. Workflows choose names from
this list only; nothing is downloaded at run time.
"""

from __future__ import annotations

import os

FOLDER = "ema_lightning"
REQUIRED = ("ema.pt", "decoder.pt")


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def _bases() -> list[str]:
    import folder_paths

    return [b for b in folder_paths.get_folder_paths(FOLDER) if os.path.isdir(b)]


def _complete(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, name)) for name in REQUIRED)


def models() -> list[str]:
    found = set()
    for base in _bases():
        found.update(entry.name for entry in os.scandir(base) if entry.is_dir() and _complete(entry.path))
    return sorted(found)


def model_path(name: str) -> str:
    """The folder of a listed name. Names come from the directory listing (no path separators); the folder is looked up again."""
    for base in _bases():
        for entry in os.scandir(base):
            if entry.name == name and entry.is_dir() and _complete(entry.path):
                return entry.path
    raise FileNotFoundError(f"model {name!r} was not found in models/{FOLDER} (see the README for the download command)")
