"""Collect results/<encoder>/<name>_seed<k>.json into results/RESULTS.md (mean ± std over seeds).

    python scripts/make_table.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import ENCODERS

ORDER = ["zeroshot", "zeroshot_desc", "zeroshot_both", "proto", "proto_aug", "fusion", "fusion_aug",
         "fusion_aug_flip", "tip_adapter", "tip_adapter_f", "linear_head", "transductive",
         "transductive_sinkhorn"]


def ms(xs):
    xs = np.array(xs) * 100
    return f"{xs.mean():.2f} ± {xs.std(ddof=1):.2f}" if len(xs) > 1 else f"{xs.mean():.2f}"


runs = defaultdict(list)
for f in sorted((ROOT / "results").glob("*/*_seed*.json")):
    r = json.loads(f.read_text())
    runs[(r["encoder"], r["name"])].append(r)

lines = ["# Results", "",
         "Accuracy in %, mean ± sample std over the seeds listed (the seed changes which training "
         "image is the single shot of each class). `val` = 480 held-out training images, used only "
         "to choose hyperparameters; `test` = the 400-image test set, evaluated once per number. "
         "`test @ defaults` uses the fixed hyperparameters in the config and no validation labels.", ""]
for enc in [*ENCODERS, "ensemble"]:
    names = [n for e, n in runs if e == enc]
    if not names:
        continue
    names = [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)
    lines += [f"## {enc}", "", "| method | seeds | val (tuned) | test (tuned) | test @ defaults | seed-0 test |",
              "|---|---|---|---|---|---|"]
    for n in names:
        rs = sorted(runs[(enc, n)], key=lambda r: r["seed"])
        s0 = next((r for r in rs if r["seed"] == 0), None)
        dflt = ms([r["default_test_acc"] for r in rs]) if "default_test_acc" in rs[0] else "n/a"
        lines.append(f"| {n} | {','.join(str(r['seed']) for r in rs)} | {ms([r['val_acc'] for r in rs])} | "
                     f"{ms([r['test_acc'] for r in rs])} | {dflt} | "
                     + (f"{s0['test_correct']}/{s0['test_n']} |" if s0 else "n/a |"))
    lines.append("")
sens = ROOT / "results/shot_sensitivity.json"
if sens.exists():
    lines += ["## Shot sensitivity (20 random shot draws, fixed hyperparameters, no validation labels)", "",
              "| encoder | method | val | test | test range |", "|---|---|---|---|---|"]
    for enc, rows in json.loads(sens.read_text()).items():
        for label, a in rows.items():
            t = np.array(a["test"]) * 100
            lines.append(f"| {enc} | {label} | {ms(a['val'])} | {ms(a['test'])} | {t.min():.2f} to {t.max():.2f} |")
    lines.append("")
(ROOT / "results/RESULTS.md").write_text("\n".join(lines))
print("\n".join(lines))
