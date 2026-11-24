#!/usr/bin/env python
"""Evaluate a saved one-shot classifier on the test (or validation) split.

    python evaluate.py --checkpoint checkpoints/clip_l14/transductive_seed0.pt
    python evaluate.py --checkpoint ... --from-images     # re-embed the images, ignore the cache

The checkpoint holds the classifier built from the 16 shots (prototypes / cache keys / prompt
weights) and its chosen hyperparameters; the frozen encoder is downloaded by OpenCLIP.
The test split is meant to be evaluated once per reported number.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import CLASSES, DATA, FEATS, ROOT, _flip_avg, list_test
from methods import METHODS


def split_embeddings(encoder, split, from_images=False, flip=False):
    cache = FEATS / f"{encoder}.pt"
    val = json.loads((ROOT / "splits/val_split.json").read_text())["val"]
    if cache.exists() and not from_images:
        ck = torch.load(cache, map_location="cpu", weights_only=False)
        if split == "test":
            return _flip_avg(ck["test"], flip), ck["test"]["labels"]
        idx = {p: i for i, p in enumerate(ck["train"]["paths"])}
        vi = torch.tensor([idx[p] for p in val])
        return _flip_avg(ck["train"], flip)[vi], ck["train"]["labels"][vi]
    from encode import Encoder
    enc = Encoder(encoder, "cuda" if torch.cuda.is_available() else "cpu")
    if split == "test":
        paths, labels = list_test()
        files = [DATA / "test" / p for p in paths]
    else:
        files = [DATA / "train" / p for p in val]
        labels = [CLASSES.index(p.split("/")[0]) for p in val]
    if not flip:
        return enc.images(files, split), torch.tensor(labels)
    feats, flipped = enc.images(files, split, flip=True)
    return _flip_avg({"feats": feats, "feats_flip": flipped}, True), torch.tensor(labels)


def standardise(scores):
    return (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True)


def predict_checkpoint(ck, split, from_images=False):
    """Class scores and labels. An ensemble checkpoint averages its members' standardised scores."""
    if ck["method"] == "ensemble":
        outs = [predict_checkpoint(m, split, from_images) for m in ck["members"]]
        return torch.stack([standardise(s) for s, _ in outs]).mean(0), outs[0][1]
    feats, labels = split_embeddings(ck["encoder"], split, from_images, ck.get("flip_tta", False))
    return METHODS[ck["method"]][1](ck["state"], ck["hp"], feats), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--from-images", action="store_true")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert ck["classes"] == CLASSES
    pred, labels = predict_checkpoint(ck, args.split, args.from_images)
    pred = pred.argmax(1)
    correct, n = int((pred == labels).sum()), len(labels)
    print(f"{ck['encoder']} / {ck['name']} / seed {ck['seed']} on {args.split}: "
          f"acc {correct / n * 100:.2f}% ({correct}/{n})")
    per_class = {}
    for i, c in enumerate(CLASSES):
        m = labels == i
        per_class[c] = float((pred[m] == i).float().mean())
        print(f"  {c:14s} {per_class[c] * 100:6.1f}%")
    if args.out:
        out = {"checkpoint": args.checkpoint, "split": args.split, "n": n, "correct": correct,
               "acc": correct / n, "per_class": per_class, "hp": ck["hp"], "shots": ck["shots"]}
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
