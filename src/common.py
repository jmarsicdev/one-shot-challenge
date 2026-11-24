"""Shared constants and data helpers: classes, prompts, encoders, splits, cached features."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATS = ROOT / "features"

CLASSES = [
    "Bedroom", "Coast", "Flower", "Forest", "Highway", "Industrial", "InsideCity", "Kitchen",
    "LivingRoom", "Mountain", "Office", "OpenCountry", "Store", "Street", "Suburb", "TallBuilding",
]

# Natural-language names for the folder labels. Written once, before any result was looked at.
CLASS_TEXT = {
    "Bedroom": "bedroom",
    "Coast": "coast",
    "Flower": "flower",
    "Forest": "forest",
    "Highway": "highway",
    "Industrial": "industrial area",
    "InsideCity": "inner city",
    "Kitchen": "kitchen",
    "LivingRoom": "living room",
    "Mountain": "mountain",
    "Office": "office",
    "OpenCountry": "open country",
    "Store": "store",
    "Street": "street",
    "Suburb": "suburb",
    "TallBuilding": "tall building",
}

TEMPLATES = [
    "a photo of a {}.",
    "a black and white photo of a {}.",
    "a grayscale photo of the {}.",
    "a photo of the {} scene.",
    "an old photograph of a {}.",
    "a picture taken at the {}.",
    "a {}.",
]

# Short visual descriptions, also written once up front; used as a second prompt set.
CLASS_DESC = {
    "Bedroom": "a bedroom with a bed and pillows",
    "Coast": "a coast with the sea, waves and a beach",
    "Flower": "a close-up of a flower with petals",
    "Forest": "a forest full of trees",
    "Highway": "a highway with lanes stretching into the distance",
    "Industrial": "an industrial site with factories, warehouses and smokestacks",
    "InsideCity": "a city center with shops and buildings along the street",
    "Kitchen": "a kitchen with cabinets, a stove and a sink",
    "LivingRoom": "a living room with a sofa and armchairs",
    "Mountain": "a mountain landscape with rocky peaks",
    "Office": "an office with a desk, a chair and a computer",
    "OpenCountry": "open countryside with fields and a wide horizon",
    "Store": "the inside of a store with shelves of goods",
    "Street": "a street lined with buildings and parked cars",
    "Suburb": "a suburban house with a lawn and a driveway",
    "TallBuilding": "a tall building, a skyscraper seen from below",
}

DESC_TEMPLATES = ["a photo of {}.", "a black and white photo of {}."]

# name -> (open_clip architecture, pretrained tag)
ENCODERS = {
    "clip_b16": ("ViT-B-16", "openai"),
    "clip_l14": ("ViT-L-14", "openai"),
    "siglip_so400m_224": ("ViT-SO400M-14-SigLIP", "webli"),
    "siglip_so400m_384": ("ViT-SO400M-14-SigLIP-384", "webli"),
    "siglip2_so400m_384": ("ViT-SO400M-16-SigLIP2-384", "webli"),
}

N_AUG = 32  # augmented views cached per shot image


def load_shots(seed):
    """Relative paths (inside data/train) of the one labelled image per class, in class order."""
    shots = json.loads((ROOT / "splits/shots.json").read_text())[str(seed)]
    return [shots[c] for c in CLASSES]


def all_shot_paths():
    shots = json.loads((ROOT / "splits/shots.json").read_text())
    return sorted({p for s in shots.values() for p in s.values()})


def list_train():
    paths, labels = [], []
    for ci, c in enumerate(CLASSES):
        for p in sorted((DATA / "train" / c).glob("*")):
            paths.append(f"{c}/{p.name}")
            labels.append(ci)
    return paths, labels


def list_test():
    paths, labels = [], []
    for ci, c in enumerate(CLASSES):
        for p in sorted((DATA / "test" / c).glob("*")):
            paths.append(f"{c}/{p.name}")
            labels.append(ci)
    return paths, labels


def _flip_avg(d, flip):
    """Query embeddings; with flip TTA, the renormalised mean of the image and its mirror."""
    if not flip:
        return d["feats"]
    return torch.nn.functional.normalize(d["feats"] + d["feats_flip"], dim=-1)


def pick_text(ck, prompts):
    """Prompt embeddings [classes, prompts, dim]: class-name templates, descriptions, or both."""
    return {"names": ck["text"], "desc": ck["text_desc"],
            "both": torch.cat([ck["text"], ck["text_desc"]], dim=1)}[prompts]


def load_context(encoder, seed, flip=False, prompts="names", shots=None):
    """Everything a method may touch for one (encoder, seed), from the embedding cache.

    shot / shot_aug : the 16 labelled images (one per class) and their augmented views
    text            : [classes, templates, dim] prompt embeddings
    val, test       : (features, labels) of the 480 validation and 400 test images
    """
    ck = torch.load(FEATS / f"{encoder}.pt", map_location="cpu", weights_only=False)
    split = json.loads((ROOT / "splits/val_split.json").read_text())
    idx = {p: i for i, p in enumerate(ck["train"]["paths"])}
    shots = shots or load_shots(seed)
    assert all(s in set(split["train"]) for s in shots), "shots must come from the training part"
    vi = torch.tensor([idx[p] for p in split["val"]])
    si = torch.tensor([idx[p] for p in shots])
    assert ck["train"]["labels"][si].tolist() == list(range(len(CLASSES)))
    return {
        "encoder": encoder,
        "seed": seed,
        "shots": shots,
        "shot": ck["train"]["feats"][si],
        "shot_aug": torch.stack([ck["shot_aug"][p] for p in shots]) if shots[0] in ck["shot_aug"] else None,
        "text": pick_text(ck, prompts),
        "scale": ck["logit_scale"],
        "val": (_flip_avg(ck["train"], flip)[vi], ck["train"]["labels"][vi]),
        "test": (_flip_avg(ck["test"], flip), ck["test"]["labels"]),
    }
