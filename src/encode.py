"""Frozen encoder wrappers: load an OpenCLIP model, embed images, augmented views and prompts."""
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from common import CLASS_DESC, CLASS_TEXT, CLASSES, DESC_TEMPLATES, ENCODERS, GRAY_ENCODERS, N_AUG, TEMPLATES


def load_image(file, gray=False):
    """RGB image; with gray=True colour is removed first (3-channel grayscale)."""
    img = Image.open(file)
    return img.convert("L").convert("RGB") if gray else img.convert("RGB")


class _Images(torch.utils.data.Dataset):
    def __init__(self, files, tf, gray=False):
        self.files, self.tf, self.gray = files, tf, gray

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(load_image(self.files[i], self.gray))


class Encoder:
    def __init__(self, name, device="cuda"):
        arch, tag = {**ENCODERS, **GRAY_ENCODERS}[name]
        self.name, self.device, self.gray = name, device, name in GRAY_ENCODERS
        model, _, self.tf = open_clip.create_model_and_transforms(arch, pretrained=tag)
        self.model = model.to(device).eval().requires_grad_(False)
        self.tok = open_clip.get_tokenizer(arch)
        self.scale = float(model.logit_scale.exp())
        size = model.visual.image_size
        size = size if isinstance(size, int) else size[0]
        # Mild augmentation for the single shot: crop, flip, brightness/contrast. The encoder's own
        # tensor conversion and normalisation (last steps of its preprocessing) are reused.
        self.aug_tf = T.Compose([
            T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            *self.tf.transforms[-3:],
        ])

    @torch.no_grad()
    def _embed(self, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            f = self.model.encode_image(x.to(self.device))
        return F.normalize(f.float(), dim=-1).cpu()

    def images(self, files, desc="", flip=False):
        """Embeddings [n, dim]; with flip=True also the embeddings of the mirrored images."""
        dl = torch.utils.data.DataLoader(_Images(files, self.tf, self.gray), batch_size=32, num_workers=8)
        out, out_flip = [], []
        for x in tqdm(dl, desc=f"{self.name} {desc}"):
            out.append(self._embed(x))
            if flip:
                out_flip.append(self._embed(x.flip(-1)))
        return (torch.cat(out), torch.cat(out_flip)) if flip else torch.cat(out)

    def augmented(self, file, seed=0):
        """N_AUG augmented views of one image -> [N_AUG, dim]. Deterministic given the seed."""
        img = load_image(file, self.gray)
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        x = torch.stack([self.aug_tf(img) for _ in range(N_AUG)])
        torch.random.set_rng_state(state)
        return self._embed(x)

    @torch.no_grad()
    def text(self, descriptions=False):
        """[classes, templates, dim] prompt embeddings (class names, or short descriptions)."""
        names, templates = (CLASS_DESC, DESC_TEMPLATES) if descriptions else (CLASS_TEXT, TEMPLATES)
        out = []
        for c in CLASSES:
            t = self.tok([tpl.format(names[c]) for tpl in templates]).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                f = self.model.encode_text(t)
            out.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.stack(out)
