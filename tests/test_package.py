"""The installed package must not contain code that reads environment variables, opens network connections or starts processes (the
categories of the Comfy Registry's automated scan), and the download manifest must cover the files the nodes load."""

import hashlib
import importlib.util
import re
from pathlib import Path

from ema_comfy import paths

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "ema_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")
spec = importlib.util.spec_from_file_location("download_model", ROOT / "tools" / "download_model.py")
download_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_model)


def test_package_has_no_environment_network_or_process_code():
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, "\n".join(hits)


def test_manifest_covers_the_model_files(model_dir):
    files = download_model.expected_files()
    assert set(paths.REQUIRED) <= set(files)
    for name, digest in files.items():
        assert hashlib.sha256((model_dir / name).read_bytes()).hexdigest() == digest, name


def test_verify_only_reports_problems(tmp_path, capsys, monkeypatch):
    folder = tmp_path / download_model.FOLDER
    folder.mkdir()
    (folder / "ema.pt").write_bytes(b"x")
    manifest = tmp_path / "m.sha256"
    manifest.write_text(f"{hashlib.sha256(b'x').hexdigest()}  ema.pt\n{'0' * 64}  decoder.pt\n", encoding="ascii")
    monkeypatch.setattr(download_model, "MANIFEST", manifest)
    assert download_model.main(["--models-dir", str(tmp_path), "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "ok       ema-lightning/ema.pt" in out and "MISSING  ema-lightning/decoder.pt" in out
