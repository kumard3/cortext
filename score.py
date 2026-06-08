"""Score text through TRIBE v2 from the command line.

Usage:
  ./score.py "your draft here"
  ./score.py "draft one" "draft two"
  ./score.py --file drafts.txt        # one draft per line
  echo "a draft" | ./score.py -        # read from stdin

CAVEAT: TRIBE predicts fMRI brain response to passively consumed media. It is
NOT a validated proxy for likes/upvotes. total_activation correlates ~0.97
with length; rank by peak / variance instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
from tribev2.demo_utils import TribeModel

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

# Override with TRIBE_DEVICE=cpu|cuda|mps. Default: CUDA if present, else CPU.
_PREF = os.environ.get("TRIBE_DEVICE", "auto").lower()
if _PREF == "cuda" or (_PREF == "auto" and torch.cuda.is_available()):
    _DEVICE = "cuda"
elif _PREF == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"
CONFIG_UPDATE = {
    "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
    "data.text_feature.device": _DEVICE,
    "data.audio_feature.device": _DEVICE,
}


def read_drafts(argv: list[str]) -> dict[str, str]:
    if argv and argv[0] == "--file":
        if len(argv) < 2:
            raise SystemExit("--file requires a path")
        texts = [l.strip() for l in Path(argv[1]).read_text().splitlines() if l.strip()]
    elif argv == ["-"]:
        texts = [l.strip() for l in sys.stdin.read().splitlines() if l.strip()]
    elif argv:
        texts = argv
    else:
        raise SystemExit(__doc__)
    return {str(i + 1): t for i, t in enumerate(texts)}


def score_one(model: TribeModel, key: str, text: str) -> dict:
    text_path = CACHE / f"score_{key}.txt"
    text_path.write_text(text)
    df = model.get_events_dataframe(text_path=text_path)
    preds, _ = model.predict(events=df)
    preds = np.asarray(preds)
    if preds.size == 0:
        return {"key": key, "text": text, "error": "no predictions"}
    return {
        "key": key,
        "text": text,
        "chars": len(text),
        "total_activation": float(preds.sum()),
        "mean_activation": float(preds.mean()),
        "peak_activation": float(preds.max()),
        "per_vertex_variance_mean": float(preds.var(axis=0).mean()),
    }


def main() -> int:
    drafts = read_drafts(sys.argv[1:])
    print(f"Loading TRIBE v2 (device={_DEVICE})...", flush=True)
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=CACHE, device=_DEVICE, config_update=CONFIG_UPDATE
    )
    print("Model loaded.\n", flush=True)

    results = []
    for key, text in drafts.items():
        print(f"=== Draft {key} ({len(text)} chars) ===\n  {text[:90]}", flush=True)
        try:
            r = score_one(model, key, text)
            results.append(r)
            if "error" in r:
                print(f"  ERROR: {r['error']}\n", flush=True)
            else:
                print(f"  peak={r['peak_activation']:.3f}  var={r['per_vertex_variance_mean']:.4f}  "
                      f"mean={r['mean_activation']:.4f}\n", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  RUN ERROR: {type(e).__name__}: {e}\n", flush=True)

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["peak_activation"], reverse=True)
    print("\n=== LEADERBOARD (peak = length-independent salience; NOT engagement) ===")
    print(f"{'rank':<5}{'peak':>8}{'var':>9}{'mean':>9}  text")
    for i, r in enumerate(ok, 1):
        print(f"{i:<5}{r['peak_activation']:>8.3f}{r['per_vertex_variance_mean']:>9.4f}"
              f"{r['mean_activation']:>9.4f}  {r['text'][:55]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
