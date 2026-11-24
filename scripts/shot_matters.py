"""Does the one shot matter?

Compares, per encoder, four choices of the single labelled image per class:

  none      no training image at all (prompts only)
  random    the three seeded random shots of splits/shots.json (mean ± std over seeds)
  typical   ORACLE: per class, the training image closest to the class mean embedding
  atypical  ORACLE: per class, the training image farthest from the class mean embedding

The two oracle choices need every training label to compute the class means. They are an analysis
of how good or bad a single shot can be, not a legal one-shot recipe, and no headline number comes
from them. All variants use the raw shot embedding (no augmented views), the combined prompt set,
and the same hyperparameter grids and selection rule as train.py (tuned on validation, test once;
plus the fixed defaults that use no validation label).

    python scripts/shot_matters.py            # writes results/shot_matters.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from common import CLASSES, ENCODERS, FEATS, load_context
from methods import METHODS, accuracy, text_proto
from train import grid

ROWS = ["tip_adapter", "transductive", "transductive_sinkhorn"]  # config names


def oracle_shots(encoder, typical):
    ck = torch.load(FEATS / f"{encoder}.pt", map_location="cpu", weights_only=False)
    train = set(json.loads((ROOT / "splits/val_split.json").read_text())["train"])
    keep = torch.tensor([p in train for p in ck["train"]["paths"]])
    feats, labels = ck["train"]["feats"], ck["train"]["labels"]
    shots = []
    for c in range(len(CLASSES)):
        idx = torch.where(keep & (labels == c))[0]
        sim = feats[idx] @ F.normalize(feats[idx].mean(0), dim=-1)
        shots.append(ck["train"]["paths"][idx[sim.argmax() if typical else sim.argmin()]])
    return shots


def run(encoder, name, shots, seed=0):
    """Tune on val, test once. `shots=None` means no training image."""
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    ctx = load_context(encoder, seed, flip=cfg.get("flip_tta", False), prompts="both",
                       shots=shots or None)
    predict = METHODS[cfg["method"]][1]
    if cfg["method"] == "tip_adapter":
        state = {"keys": ctx["shot"], "txt": text_proto(ctx), "scale": ctx["scale"]}
    else:
        state = {"img": ctx["shot"] if shots else None, "txt": text_proto(ctx)}
    default = cfg["default"]
    (xv, yv), (xt, yt) = ctx["val"], ctx["test"]
    best_hp, best_val = default, -1.0
    for hp in grid(cfg):
        hp = {**default, **hp}
        v = accuracy(predict(state, hp, xv), yv)
        if v > best_val:
            best_hp, best_val = hp, v
    return {"val": best_val, "test": accuracy(predict(state, best_hp, xt), yt), "hp": best_hp,
            "default_val": accuracy(predict(state, default, xv), yv),
            "default_test": accuracy(predict(state, default, xt), yt)}


def main():
    all_shots = json.loads((ROOT / "splits/shots.json").read_text())
    out = {}
    for enc in ENCODERS:
        variants = {
            "none": [None],
            "random": [[all_shots[s][c] for c in CLASSES] for s in ("0", "1", "2")],
            "typical": [oracle_shots(enc, True)],
            "atypical": [oracle_shots(enc, False)],
        }
        out[enc] = {}
        for vname, shot_sets in variants.items():
            for row in ROWS:
                if vname == "none" and row == "tip_adapter":
                    continue  # Tip-Adapter without a shot is plain zero-shot
                runs = [run(enc, row, shots, seed) for seed, shots in enumerate(shot_sets)]
                out[enc][f"{vname}/{row}"] = runs
                t = [r["test"] * 100 for r in runs]
                print(f"{enc:20s} {vname:9s} {row:22s} val {sum(r['val'] for r in runs)/len(runs)*100:5.2f} "
                      f"test {sum(t)/len(t):5.2f}  defaults test "
                      f"{sum(r['default_test'] for r in runs)/len(runs)*100:5.2f}")
        out[enc]["shots"] = {"typical": variants["typical"][0], "atypical": variants["atypical"][0]}
    (ROOT / "results/shot_matters.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
