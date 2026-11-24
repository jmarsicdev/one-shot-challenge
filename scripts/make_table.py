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
from common import ENCODERS, GRAY_ENCODERS

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
GRAY_ROWS = ["zeroshot_both", "tip_adapter_f", "transductive", "transductive_sinkhorn"]
if any(e in GRAY_ENCODERS for e, _ in runs):
    lines += ["## Grayscale ablation", "",
              "Only the Flower class is stored in colour (all 175 Flower images, train and test); the other 15 "
              "classes are grayscale files. Here every image is converted to 3-channel grayscale before the "
              "encoder, so colour cannot act as a shortcut for Flower. Same shots, grids and selection rule; "
              "mean ± std over seeds 0, 1, 2.", "",
              "| encoder | method | val, original | val, grayscale | test, original | test, grayscale |",
              "|---|---|---|---|---|---|"]
    for enc in ENCODERS:
        for n in GRAY_ROWS:
            a, b = runs.get((enc, n)), runs.get((f"{enc}_gray", n))
            if a and b:
                lines.append(f"| {enc} | {n} | {ms([r['val_acc'] for r in a])} | {ms([r['val_acc'] for r in b])} | "
                             f"{ms([r['test_acc'] for r in a])} | {ms([r['test_acc'] for r in b])} |")
    lines += ["", "Reading: removing colour changes the Flower embeddings (cosine to the colour version about "
              "0.87 to 0.90) but not the predictions; accuracy is unchanged to within one image. Colour is "
              "not what makes Flower easy, and the other 15 classes never had colour.", ""]
sm = ROOT / "results/shot_matters.json"
if sm.exists():
    lines += ["## Does the one shot matter?", "",
              "`none` = no training image (prompts only); `random` = the three seeded shots (mean ± std); "
              "`typical` / `atypical` = ORACLE choices, the training image closest to / farthest from its class "
              "mean embedding. The oracle rows need all training labels and are analysis only, never a "
              "reported one-shot result. Raw shot embedding (no augmented views), so `random` differs slightly "
              "from the main tables. Hyperparameters tuned on val as everywhere else; `test @ defaults` uses "
              "no validation label. See `scripts/shot_matters.py`.", "",
              "| encoder | shot | method | val (tuned) | test (tuned) | test @ defaults |", "|---|---|---|---|---|---|"]
    for enc, rows in json.loads(sm.read_text()).items():
        for key, rs in rows.items():
            if key == "shots":
                continue
            v, m = key.split("/")
            lines.append(f"| {enc} | {v} | {m} | {ms([r['val'] for r in rs])} | {ms([r['test'] for r in rs])} | "
                         f"{ms([r['default_test'] for r in rs])} |")
    lines += ["", "Reading: for the inductive Tip-Adapter the shot matters (typical shots beat random ones, "
              "atypical ones are worse). For transductive refinement, validation accuracy is about the same "
              "with no shot at all as with a random shot: the unlabelled queries and the prompts do the work. "
              "The single-run test numbers without a shot or with an oracle shot are less stable: in a few "
              "cases the setting chosen on validation merges two similar classes on the 400-image test batch "
              "and test accuracy drops to 87 to 90 %, which the three random-shot seeds never did.", ""]
(ROOT / "results/RESULTS.md").write_text("\n".join(lines))
print("\n".join(lines))
