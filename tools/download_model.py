"""Download the official EMA Lightning model into ComfyUI's models/ema_lightning and verify it.

    python tools/download_model.py --models-dir <ComfyUI>/models/ema_lightning [--verify-only]

Source: https://huggingface.co/canberkkkkkk/ema-lightning, pinned to the revision this node pack was verified with (about 34 MB:
ema.pt, decoder.pt, config.json; Apache-2.0). The files land in models/ema_lightning/ema-lightning/ and are checked against the
SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with ema-lightning). Nothing is downloaded unless you run
this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "canberkkkkkk/ema-lightning"
REVISION = "7a6ba1ad216bb2f1da9863f80ac8770a6a807632"
FOLDER = "ema-lightning"
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{file name: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/ema_lightning folder")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    files = expected_files()
    target = args.models_dir / FOLDER
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=REPO, revision=REVISION, local_dir=str(target), allow_patterns=sorted(files))
    ok = True
    for name, digest in files.items():
        path = target / name
        good = path.is_file() and sha256(path) == digest
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if path.is_file() else 'MISSING ')} {FOLDER}/{name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
