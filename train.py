#!/usr/bin/env python
"""Fit one one-shot method for one encoder and seed, from cached embeddings.

    python train.py --config configs/tip_adapter_f.yaml --encoder clip_l14 --seed 0
    python train.py --config configs/fusion.yaml --all        # every encoder, seeds 0 1 2

Steps: build the classifier from the 16 shots (one per class) and the prompts, choose the
inference hyperparameters on the 480-image validation split, then evaluate the test set once with
the chosen setting. Also recorded: accuracy at the config's fixed `default` hyperparameters, which
uses no validation labels at all.

Writes checkpoints/<encoder>/<name>_seed<k>.pt and results/<encoder>/<name>_seed<k>.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import CLASSES, ENCODERS, GRAY_ENCODERS, ROOT, load_context
from methods import METHODS, accuracy


def expand(spec):
    if isinstance(spec, dict):
        (kind, (lo, hi, n)), = spec.items()
        vals = np.linspace(lo, hi, n) if kind == "linspace" else np.geomspace(lo, hi, n)
        return [round(float(v), 6) for v in vals]
    return list(spec)


def grid(cfg):
    g = {k: expand(v) for k, v in (cfg.get("grid") or {}).items()}
    return [dict(zip(g, vals)) for vals in itertools.product(*g.values())]


def run(cfg, encoder, seed):
    torch.manual_seed(seed)
    ctx = load_context(encoder, seed, flip=cfg.get("flip_tta", False), prompts=cfg.get("prompts", "names"))
    prepare, predict = METHODS[cfg["method"]]
    state = prepare(ctx, cfg)
    (xv, yv), (xt, yt) = ctx["val"], ctx["test"]

    default = cfg.get("default") or {}
    best_hp, best_val = default, -1.0
    for hp in grid(cfg) or [default]:
        hp = {**default, **hp}
        v = accuracy(predict(state, hp, xv), yv)
        if v > best_val:  # first best in grid order wins ties
            best_hp, best_val = hp, v

    test_scores = predict(state, best_hp, xt)  # the one test evaluation of the tuned model
    test_acc = accuracy(test_scores, yt)
    res = {
        "name": cfg["name"], "method": cfg["method"], "encoder": encoder, "seed": seed,
        "shots": ctx["shots"], "hp": best_hp, "flip_tta": cfg.get("flip_tta", False),
        "val_acc": best_val, "test_acc": test_acc,
        "test_correct": int((test_scores.argmax(1) == yt).sum()), "test_n": len(yt),
        "default_hp": default,
        "default_val_acc": accuracy(predict(state, default, xv), yv),
        "default_test_acc": accuracy(predict(state, default, xt), yt),
    }
    ck = {**res, "state": state, "config": cfg, "classes": CLASSES}
    for folder, payload in (("checkpoints", ck), ("results", res)):
        d = ROOT / folder / encoder
        d.mkdir(parents=True, exist_ok=True)
        stem = d / f"{cfg['name']}_seed{seed}"
        if folder == "checkpoints":
            torch.save(payload, stem.with_suffix(".pt"))
        else:
            stem.with_suffix(".json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{encoder:14s} {cfg['name']:18s} seed {seed}: val {best_val*100:5.2f}  "
          f"test {test_acc*100:5.2f} ({res['test_correct']}/{res['test_n']})  "
          f"[defaults: val {res['default_val_acc']*100:5.2f} test {res['default_test_acc']*100:5.2f}]  hp {best_hp}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, nargs="+")
    ap.add_argument("--encoder", default="clip_l14", choices=[*ENCODERS, *GRAY_ENCODERS])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every encoder, seeds 0 1 2")
    ap.add_argument("--all-gray", action="store_true", help="every grayscale-ablation encoder, seeds 0 1 2")
    args = ap.parse_args()
    for path in args.config:
        cfg = yaml.safe_load(Path(path).read_text())
        cfg.setdefault("name", Path(path).stem)
        runs = [(args.encoder, args.seed)]
        if args.all or args.all_gray:
            runs = itertools.product(GRAY_ENCODERS if args.all_gray else ENCODERS, [0, 1, 2])
        for encoder, seed in runs:
            run(cfg, encoder, seed)


if __name__ == "__main__":
    main()
