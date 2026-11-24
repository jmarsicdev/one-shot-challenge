"""How much does the result depend on which image is the shot?

20 random shot draws (numpy.random.RandomState(seed), seeds 0..19; one image per class from the
training part of the split) evaluated with training-free methods at fixed hyperparameters, so no
validation label is involved.

    python scripts/shot_sensitivity.py            # writes results/shot_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import CLASSES, ENCODERS, load_context
from methods import METHODS, accuracy

DRAWS = 20
ROWS = {  # label -> (method, fixed hyperparameters)
    "proto": ("proto", {}),
    "tip_adapter (alpha 1, beta 5.5)": ("tip_adapter", {"alpha": 1.0, "beta": 5.5}),
}

split = json.loads((ROOT / "splits/val_split.json").read_text())
by_class = {c: sorted(p for p in split["train"] if p.startswith(c + "/")) for c in CLASSES}
out = {}
for enc in ENCODERS:
    accs = {k: {"val": [], "test": []} for k in ROWS}
    for seed in range(DRAWS):
        rs = np.random.RandomState(seed)
        shots = [by_class[c][rs.randint(len(by_class[c]))] for c in CLASSES]
        ctx = load_context(enc, seed, prompts="both", shots=shots)
        for label, (method, hp) in ROWS.items():
            prepare, predict = METHODS[method]
            state = prepare(ctx, {})
            for s in ("val", "test"):
                accs[label][s].append(accuracy(predict(state, hp, ctx[s][0]), ctx[s][1]))
    out[enc] = accs
    for label, a in accs.items():
        t = np.array(a["test"]) * 100
        print(f"{enc:20s} {label:32s} test {t.mean():5.2f} ± {t.std(ddof=1):4.2f}  "
              f"range {t.min():5.2f} - {t.max():5.2f}")
(ROOT / "results/shot_sensitivity.json").write_text(json.dumps(out, indent=1) + "\n")
