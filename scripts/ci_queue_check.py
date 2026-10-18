"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server with this node pack installed and run the
shipped workflows through POST /prompt with the real model (34 MB) on the CPU.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --model <folder with ema.pt and decoder.pt> [--workdir DIR] [--keep-pack]

The workflows are run with `device` set to cpu. This proves that the pack installs into a stock ComfyUI, that Loader -> Text to
Speech -> Save Audio (Advanced) executes through the queue in FLAC and MP3, that the same seed gives the same audio and another seed
different audio, and that word times come back. The GPU runs are in docs/VERIFICATION.md. The server is started and stopped by this
script only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_client  # noqa: E402

NODE_IDS = {"EMALightningLoader", "EMALightningSay"}


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def decoded(path: Path):
    import av
    import numpy as np

    with av.open(str(path)) as c:
        return np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(audio=0)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-pack", action="store_true", help="use the pack already installed in custom_nodes (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="ema-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = comfy / "custom_nodes" / "ComfyUI-EMALightning"
    if pack.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(pack):  # never delete through a link into a working copy
        raise SystemExit(f"{pack} is a link; use a ComfyUI checkout without it")
    if not args.keep_pack:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--keep-pack: {pack} is not installed")
    models = work / "models" / "ema_lightning" / "ema-lightning"  # the name the shipped workflows use
    models.mkdir(parents=True, exist_ok=True)
    for name in ("ema.pt", "decoder.pt"):
        shutil.copyfile(args.model / name, models / name)
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"ema_ci:\n  base_path: {(work / 'models').as_posix()}\n  ema_lightning: ema_lightning\n", encoding="utf-8")
    output = work / "output"

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
         "--extra-model-paths-config", str(yaml), "--output-directory", str(output)],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")

        def run(name: str, prefix: str, **overrides) -> dict:
            wf = json.loads((ROOT / "workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
            wf["1"]["inputs"]["device"] = "cpu"
            wf["3"]["inputs"]["filename_prefix"] = f"ci/{prefix}"
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            result = queue_client.run(server, wf, None, 600)
            error = result.get("error") or {}
            print(json.dumps({"workflow": name, "run": prefix, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:300] if isinstance(error, dict) else error}, ensure_ascii=False))
            if result["status"] != "success":
                failures.append(f"{prefix}: {error}")
            return result

        def saved(prefix: str, ext: str):
            files = sorted((output / "ci").glob(f"{prefix}*.{ext}"))
            if len(files) != 1:
                failures.append(f"{prefix}: expected one .{ext}, found {files}")
                return None
            return files[0]

        a = run("tts_basic", "seed0_a")
        run("tts_basic", "seed1", **{"2.seed": 1})
        run("tts_basic", "seed0_b")  # the seed changed in between, so this executes again
        fa, fb, f1 = saved("seed0_a", "flac"), saved("seed0_b", "flac"), saved("seed1", "flac")
        if fa and fb and f1:
            x, y, z = decoded(fa), decoded(fb), decoded(f1)
            print("samples:", x.size)
            if not (x.shape == y.shape and (x == y).all()):
                failures.append("the same seed gave different audio")
            if x.shape == z.shape and (x == z).all():
                failures.append("different seeds gave the same audio")
        words = json.loads(((a.get("text") or {}).get("4") or ["[]"])[0])
        if not words or not all(w["end"] > w["start"] for w in words):
            failures.append(f"unexpected word times {words[:3]}")
        m = run("tts_announcement_mp3", "announcement")
        if saved("announcement", "mp3") is None or not (m.get("text") or {}).get("5"):
            failures.append("announcement: no mp3 or no report")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
