"""Backbone ensemble: average the standardised class scores of one method across encoders.

    python scripts/ensemble.py --name transductive --encoders clip_l14 siglip_so400m_384 --seed 0

Members are the already-trained checkpoints checkpoints/<encoder>/<name>_seed<k>.pt; nothing is
re-tuned. Writes checkpoints/ensemble/<tag>_seed<k>.pt (loadable by evaluate.py) and the matching
results JSON.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from common import CLASSES
from evaluate import predict_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="method config name of the members")
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    members = [torch.load(ROOT / "checkpoints" / e / f"{args.name}_seed{args.seed}.pt",
                          map_location="cpu", weights_only=False) for e in args.encoders]
    tag = args.tag or f"{args.name}__{'+'.join(args.encoders)}"
    ck = {"method": "ensemble", "name": tag, "encoder": "ensemble", "seed": args.seed,
          "members": members, "hp": {}, "shots": members[0]["shots"], "classes": CLASSES}
    res = {"name": tag, "method": "ensemble", "encoder": "ensemble", "seed": args.seed,
           "members": [f"{m['encoder']}/{m['name']}" for m in members], "shots": ck["shots"]}
    for split in ("val", "test"):
        scores, labels = predict_checkpoint(ck, split)
        correct = int((scores.argmax(1) == labels).sum())
        res[f"{split}_acc"] = correct / len(labels)
        if split == "test":
            res["test_correct"], res["test_n"] = correct, len(labels)
    ck.update({k: v for k, v in res.items() if k.endswith("acc") or k.startswith("test_")})
    for folder in ("checkpoints", "results"):
        (ROOT / folder / "ensemble").mkdir(parents=True, exist_ok=True)
    torch.save(ck, ROOT / "checkpoints/ensemble" / f"{tag}_seed{args.seed}.pt")
    (ROOT / "results/ensemble" / f"{tag}_seed{args.seed}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{tag} seed {args.seed}: val {res['val_acc']*100:.2f}  test {res['test_acc']*100:.2f} "
          f"({res['test_correct']}/{res['test_n']})")


if __name__ == "__main__":
    main()
