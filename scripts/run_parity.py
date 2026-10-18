"""Speak a fixed set of texts through the nodes, in-process in a ComfyUI environment, and save every waveform for a comparison with
the official package's `EMA().say()` (maintainer tool; the official side is run separately).

    python scripts/run_parity.py --comfyui <ComfyUI checkout> --model <models/ema_lightning/ema-lightning> --out <file.npz> [--device cuda]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = [  # (id, text, seed, speed, sample_rate, batch_size)
    ("short", "Merhaba, size nasıl yardımcı olabilirim?", 0, 1.0, 48000, "auto"),
    ("numbers", "15 Ekim 2026 Çarşamba günü saat 14:30'da, 1.250.000 TL tutarındaki ödemeniz hesabınıza yatırılacak.", 0, 1.0, 48000, "auto"),
    ("long", "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyor, martılar ise teknelerin etrafında "
             "dönerek şanslarını deniyordu. Biraz sonra ilk tekne denize açıldı ve gün yavaş yavaş aydınlandı.", 7, 1.0, 48000, "auto"),
    ("seed_42", "Merhaba, size nasıl yardımcı olabilirim?", 42, 1.0, 48000, "auto"),
    ("fast", "Bugün hava çok güzel, yarın da yağmur yağacakmış.", 3, 1.5, 48000, "auto"),
    ("slow_24k", "Bugün hava çok güzel, yarın da yağmur yağacakmış.", 3, 0.75, 24000, "auto"),
    ("16k", "Toplantı öğleden sonra başlayacak.", 5, 1.0, 16000, "auto"),
    ("batch_1", "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyor.", 9, 1.0, 48000, "1"),
    ("batch_8", "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyor.", 9, 1.0, 48000, "8"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfyui", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.comfyui))
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import folder_paths
    import nodes

    folder_paths.add_model_folder_path("ema_lightning", str(args.model.parent), is_default=True)
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    m = nodes.NODE_CLASS_MAPPINGS
    out, reports = {}, {}
    for cid, text, seed, speed, rate, batch in CASES:
        handle = m["EMALightningLoader"].execute(model=args.model.name, device=args.device, batch_size=batch, lightning=False).args[0]
        audio, words, report = m["EMALightningSay"].execute(model=handle, text=text, seed=seed, speed=speed, sample_rate=str(rate)).args
        out[cid] = audio["waveform"].reshape(-1).numpy()
        reports[cid] = {"report": json.loads(report), "words": json.loads(words)}
        print(json.dumps({"case": cid, "samples": int(out[cid].size), **{k: reports[cid]["report"][k] for k in ("duration", "batch_size", "seconds")}}),
              flush=True)
    np.savez(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps({"cases": CASES, "results": reports}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
