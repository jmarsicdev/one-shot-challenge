"""Pick the single labelled training image per class, for each seed.

Rule: take the training part of splits/val_split.json (1,920 images, validation images excluded),
sort each class's file names, and for seed s draw one index per class, in class order, from
numpy.random.default_rng(s). The result is written to splits/shots.json and never edited by hand.

    python scripts/make_shots.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import CLASSES, ROOT

SEEDS = [0, 1, 2]

split = json.loads((ROOT / "splits/val_split.json").read_text())
out = {}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    shots = {}
    for c in CLASSES:
        files = sorted(p for p in split["train"] if p.startswith(c + "/"))
        shots[c] = files[int(rng.integers(len(files)))]
    out[str(seed)] = shots
(ROOT / "splits/shots.json").write_text(json.dumps(out, indent=1) + "\n")
for s, v in out.items():
    print(s, list(v.values())[:4], "...")
