#!/usr/bin/env python
"""Cache frozen-encoder embeddings to features/<encoder>.pt.

    python extract.py                 # all encoders in src/common.py
    python extract.py clip_l14        # one encoder

Cached: embeddings of data/train (2,400) and data/test (400) and of their mirror images, prompt
embeddings for every (class, template) and for the class descriptions, and N_AUG augmented views of each shot image listed in splits/shots.json.
No label is used here beyond folder names for bookkeeping.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import CLASSES, DATA, ENCODERS, FEATS, TEMPLATES, all_shot_paths, list_test, list_train
from encode import Encoder


def main(name):
    enc = Encoder(name)
    out = {"encoder": name, "open_clip": ENCODERS[name], "classes": CLASSES, "templates": TEMPLATES,
           "logit_scale": enc.scale}
    for split, (paths, labels) in {"train": list_train(), "test": list_test()}.items():
        feats, flipped = enc.images([DATA / split / p for p in paths], split, flip=True)
        out[split] = {"paths": paths, "labels": torch.tensor(labels), "feats": feats, "feats_flip": flipped}
    out["text"] = enc.text()
    out["text_desc"] = enc.text(descriptions=True)
    out["shot_aug"] = {p: enc.augmented(DATA / "train" / p) for p in all_shot_paths()}
    FEATS.mkdir(exist_ok=True)
    torch.save(out, FEATS / f"{name}.pt")
    print(name, tuple(out["train"]["feats"].shape), "logit scale", round(enc.scale, 2))


if __name__ == "__main__":
    for n in sys.argv[1:] or list(ENCODERS):
        main(n)
        torch.cuda.empty_cache()
