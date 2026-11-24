# One-Shot Challenge

ITCS 6169/8169 Computer Vision, Assignment 3: 16-class scene recognition with **exactly one
labelled training image per class**, using frozen vision-language foundation models
(OpenCLIP, SigLIP, SigLIP 2). No encoder weight is ever updated; everything that is learned fits in
a few hundred kilobytes.

## Headline result

| | test accuracy | notes |
|---|---|---|
| **Best single checkpoint** (selected by validation accuracy on seed 0) | **97.00 % (388/400)** | SigLIP SO400M/14 @ 384, transductive prototype refinement with Sinkhorn class balancing. Sibling seeds: 385/400 and 388/400 (mean 96.75 ± 0.43 %). |
| Best without the class-balance prior | 94.75 % (379/400) | SigLIP SO400M/14 @ 224, transductive refinement. Sibling seeds: 379/400 and 381/400. |
| Best with no test-time adaptation | 92.25 % (369/400) | SigLIP 2 SO400M/16 @ 384, Tip-Adapter-F. Sibling seeds: 370/400 and 372/400. |
| Zero-shot reference (no training image at all) | 88.25 % (353/400) | SigLIP SO400M/14 @ 384, prompt ensemble. |

The full ablation (5 encoders × 13 methods × 3 seeds, backbone ensembles, shot sensitivity) is in
[`results/RESULTS.md`](results/RESULTS.md).

Things to know when reading these numbers:

- **Seeds.** The seed changes *which* training image is the single shot of each class. All
  mean ± std figures are over seeds 0, 1, 2.
- **Validation labels.** The 480-image validation split (30 per class, held out from the training
  folder, same split as the earlier challenges) is used only to choose a handful of inference
  hyperparameters (a fusion weight, Tip-Adapter's alpha/beta, the refinement temperature) and the
  prompt set. Because those images are labelled, every result file also carries the accuracy at
  **fixed default hyperparameters, which uses no validation label at all**
  (`test @ defaults` column). For the headline configuration that number is 91.00 % (364/400).
- **Test-time adaptation.** The transductive rows refine the class prototypes on the *unlabelled*
  test embeddings (the whole 400-image batch at once). Test labels are never used. The Sinkhorn
  variant additionally assumes the query set is class-balanced, as the training set is.
- **Val/test gap.** Test accuracy is usually 1 to 3 points above validation accuracy, for
  most methods and encoders (zero-shot included), so the gap looks like a property of the two image
  sets rather than of tuning.

## Method

All methods work on cached, L2-normalised embeddings from a frozen encoder.

1. **Zero-shot** – cosine similarity to a prompt ensemble: 7 class-name templates (including
   "a black and white photo of a {}", the images are grayscale) and 2 templates over a short
   description of each class; the two sets are averaged with equal weight.
2. **One-shot image prototype** – cosine similarity to the single shot's embedding.
3. **Text + image fusion** – weighted sum of the two similarities.
4. **Tip-Adapter / Tip-Adapter-F** – key-value cache over the 16 shots added to the zero-shot
   logits; in the -F variant the cache keys are fine-tuned on augmented views of the shots.
5. **Augmented shot** – the prototype is the mean embedding of the shot and 32 augmented views
   (random resized crop, flip, brightness/contrast); also a **linear head** initialised at the
   prompt weights and trained on those views with an L2 pull back toward the prompts.
6. **Transductive refinement (test-time adaptation)** – flip-averaged query embeddings, then soft
   k-means in image space: prototypes start at the augmented shot, queries are softly assigned
   using image-prototype similarity plus the text similarity, and prototypes are re-estimated from
   the queries (the shot counts as `kappa` pseudo-observations). Ablation: Sinkhorn class balancing
   of the soft assignments.
7. **Backbone ensemble** – average of the standardised class scores of several encoders.

## Layout

```
extract.py            cache embeddings of a frozen encoder -> features/<encoder>.pt
train.py              fit one method (config) for one encoder and seed, tune on val, test once
evaluate.py           evaluate a saved checkpoint (optionally re-embedding the raw images)
configs/              one YAML per method (hyperparameter grid and fixed defaults)
src/                  common.py (classes, prompts, encoders), encode.py, methods.py
scripts/              make_shots.py, ensemble.py, shot_sensitivity.py, make_table.py
splits/               val_split.json (fixed 1,920 / 480 split), shots.json (the shots per seed)
checkpoints/          trained classifiers: checkpoints/<encoder>/<method>_seed<k>.pt
results/              per-run JSON, RESULTS.md, final/test_metrics.json
```

## How the single shot is chosen

`scripts/make_shots.py`: take the training part of `splits/val_split.json` (validation images
excluded), sort each class's file names, and for seed `s` draw one index per class, in class order,
from `numpy.random.default_rng(s)`. The 16 chosen files per seed are listed in `splits/shots.json`.
No other training image, labelled or unlabelled, is used to build a classifier.

## Reproduce

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU (about 6 GB of memory is
enough; a CPU works but is slow).

```bash
uv sync
ln -s /path/to/dataset data        # must contain data/train/<Class>/*.jpg and data/test/<Class>/*.jpg
```

**Headline number from the shipped checkpoint** (downloads the SigLIP weights, embeds the 400 test
images, prints `acc 97.00% (388/400)`):

```bash
uv run python evaluate.py --checkpoint checkpoints/siglip_so400m_384/transductive_sinkhorn_seed0.pt --from-images
```

**Everything from scratch:**

```bash
uv run python scripts/make_shots.py                    # splits/shots.json (already committed)
uv run python extract.py                               # features/*.pt for the five encoders
uv run python train.py --config configs/*.yaml --all   # all methods, encoders, seeds
uv run python scripts/ensemble.py --name transductive --seed 0 \
    --encoders clip_b16 clip_l14 siglip_so400m_224 siglip_so400m_384 siglip2_so400m_384 --tag transductive__all5
uv run python scripts/shot_sensitivity.py
uv run python scripts/make_table.py                    # results/RESULTS.md
```

A single run: `uv run python train.py --config configs/tip_adapter_f.yaml --encoder clip_l14 --seed 0`.
Once the embeddings are cached, the whole table takes about half a minute on CPU.

## Encoders

| name | OpenCLIP model | pretrained |
|---|---|---|
| `clip_b16` | ViT-B-16 | openai |
| `clip_l14` | ViT-L-14 | openai |
| `siglip_so400m_224` | ViT-SO400M-14-SigLIP | webli |
| `siglip_so400m_384` | ViT-SO400M-14-SigLIP-384 | webli |
| `siglip2_so400m_384` | ViT-SO400M-16-SigLIP2-384 | webli |
