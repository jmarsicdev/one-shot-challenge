# Results

Accuracy in %, mean ± sample std over the seeds listed (the seed changes which training image is the single shot of each class). `val` = 480 held-out training images, used only to choose hyperparameters; `test` = the 400-image test set, evaluated once per number. `test @ defaults` uses the fixed hyperparameters in the config and no validation labels.

## clip_b16

| method | seeds | val (tuned) | test (tuned) | test @ defaults | seed-0 test |
|---|---|---|---|---|---|
| zeroshot | 0,1,2 | 82.08 ± 0.00 | 84.00 ± 0.00 | 84.00 ± 0.00 | 336/400 |
| zeroshot_desc | 0,1,2 | 85.42 ± 0.00 | 85.50 ± 0.00 | 85.50 ± 0.00 | 342/400 |
| zeroshot_both | 0,1,2 | 85.42 ± 0.00 | 87.00 ± 0.00 | 87.00 ± 0.00 | 348/400 |
| proto | 0,1,2 | 64.93 ± 7.35 | 65.42 ± 6.50 | 65.42 ± 6.50 | 236/400 |
| proto_aug | 0,1,2 | 69.86 ± 4.07 | 70.83 ± 4.91 | 70.83 ± 4.91 | 271/400 |
| fusion | 0,1,2 | 86.53 ± 0.43 | 88.92 ± 1.81 | 80.42 ± 5.31 | 351/400 |
| fusion_aug | 0,1,2 | 87.08 ± 0.72 | 89.17 ± 1.66 | 83.42 ± 4.00 | 349/400 |
| fusion_aug_flip | 0,1,2 | 87.50 ± 0.95 | 89.42 ± 1.46 | 83.50 ± 3.63 | 351/400 |
| tip_adapter | 0,1,2 | 87.22 ± 0.52 | 88.67 ± 2.13 | 87.42 ± 0.14 | 346/400 |
| tip_adapter_f | 0,1,2 | 87.71 ± 0.75 | 89.58 ± 1.81 | 89.83 ± 1.42 | 350/400 |
| linear_head | 0,1,2 | 87.92 ± 0.91 | 90.33 ± 1.01 | 90.08 ± 1.23 | 357/400 |
| transductive | 0,1,2 | 92.78 ± 0.32 | 92.00 ± 0.43 | 92.17 ± 0.29 | 367/400 |
| transductive_sinkhorn | 0,1,2 | 94.10 ± 0.24 | 94.42 ± 0.14 | 91.92 ± 0.38 | 377/400 |

## clip_l14

| method | seeds | val (tuned) | test (tuned) | test @ defaults | seed-0 test |
|---|---|---|---|---|---|
| zeroshot | 0,1,2 | 82.92 ± 0.00 | 85.75 ± 0.00 | 85.75 ± 0.00 | 343/400 |
| zeroshot_desc | 0,1,2 | 86.67 ± 0.00 | 90.25 ± 0.00 | 90.25 ± 0.00 | 361/400 |
| zeroshot_both | 0,1,2 | 86.25 ± 0.00 | 88.75 ± 0.00 | 88.75 ± 0.00 | 355/400 |
| proto | 0,1,2 | 64.51 ± 4.81 | 66.08 ± 1.51 | 66.08 ± 1.51 | 258/400 |
| proto_aug | 0,1,2 | 70.97 ± 3.75 | 70.17 ± 3.56 | 70.17 ± 3.56 | 271/400 |
| fusion | 0,1,2 | 86.94 ± 0.87 | 90.42 ± 1.46 | 83.50 ± 4.26 | 355/400 |
| fusion_aug | 0,1,2 | 87.22 ± 0.94 | 90.33 ± 1.63 | 84.25 ± 5.00 | 355/400 |
| fusion_aug_flip | 0,1,2 | 87.43 ± 0.84 | 90.08 ± 1.01 | 84.42 ± 5.14 | 356/400 |
| tip_adapter | 0,1,2 | 87.08 ± 0.95 | 90.83 ± 1.81 | 89.17 ± 0.38 | 355/400 |
| tip_adapter_f | 0,1,2 | 88.82 ± 0.94 | 91.92 ± 1.04 | 91.17 ± 0.80 | 363/400 |
| linear_head | 0,1,2 | 88.89 ± 0.79 | 92.17 ± 0.80 | 92.08 ± 1.51 | 365/400 |
| transductive | 0,1,2 | 92.99 ± 0.52 | 95.75 ± 0.00 | 92.92 ± 0.76 | 383/400 |
| transductive_sinkhorn | 0,1,2 | 95.07 ± 0.24 | 96.25 ± 0.00 | 94.25 ± 0.00 | 385/400 |

## siglip_so400m_224

| method | seeds | val (tuned) | test (tuned) | test @ defaults | seed-0 test |
|---|---|---|---|---|---|
| zeroshot | 0,1,2 | 81.25 ± 0.00 | 87.25 ± 0.00 | 87.25 ± 0.00 | 349/400 |
| zeroshot_desc | 0,1,2 | 88.96 ± 0.00 | 89.75 ± 0.00 | 89.75 ± 0.00 | 359/400 |
| zeroshot_both | 0,1,2 | 88.13 ± 0.00 | 90.25 ± 0.00 | 90.25 ± 0.00 | 361/400 |
| proto | 0,1,2 | 75.14 ± 5.67 | 76.75 ± 2.61 | 76.75 ± 2.61 | 300/400 |
| proto_aug | 0,1,2 | 77.85 ± 5.53 | 79.25 ± 3.25 | 79.25 ± 3.25 | 302/400 |
| fusion | 0,1,2 | 89.86 ± 0.43 | 92.17 ± 0.88 | 88.17 ± 2.96 | 372/400 |
| fusion_aug | 0,1,2 | 89.86 ± 0.87 | 91.92 ± 1.13 | 89.00 ± 3.19 | 368/400 |
| fusion_aug_flip | 0,1,2 | 89.93 ± 0.79 | 91.58 ± 1.61 | 88.83 ± 3.30 | 369/400 |
| tip_adapter | 0,1,2 | 89.86 ± 0.32 | 92.00 ± 1.15 | 90.50 ± 0.00 | 372/400 |
| tip_adapter_f | 0,1,2 | 90.28 ± 1.15 | 92.00 ± 0.43 | 91.58 ± 0.63 | 366/400 |
| linear_head | 0,1,2 | 89.72 ± 0.94 | 90.67 ± 1.46 | 90.50 ± 1.15 | 356/400 |
| transductive | 0,1,2 | 94.72 ± 0.24 | 94.92 ± 0.29 | 92.83 ± 0.88 | 379/400 |
| transductive_sinkhorn | 0,1,2 | 95.56 ± 0.32 | 97.00 ± 0.43 | 93.33 ± 0.14 | 389/400 |

## siglip_so400m_384

| method | seeds | val (tuned) | test (tuned) | test @ defaults | seed-0 test |
|---|---|---|---|---|---|
| zeroshot | 0,1,2 | 82.08 ± 0.00 | 84.75 ± 0.00 | 84.75 ± 0.00 | 339/400 |
| zeroshot_desc | 0,1,2 | 87.71 ± 0.00 | 88.25 ± 0.00 | 88.25 ± 0.00 | 353/400 |
| zeroshot_both | 0,1,2 | 86.25 ± 0.00 | 88.25 ± 0.00 | 88.25 ± 0.00 | 353/400 |
| proto | 0,1,2 | 73.96 ± 6.16 | 75.25 ± 6.13 | 75.25 ± 6.13 | 277/400 |
| proto_aug | 0,1,2 | 76.32 ± 6.37 | 78.92 ± 3.82 | 78.92 ± 3.82 | 299/400 |
| fusion | 0,1,2 | 89.03 ± 0.60 | 90.42 ± 1.42 | 88.58 ± 2.70 | 357/400 |
| fusion_aug | 0,1,2 | 89.31 ± 0.98 | 91.17 ± 1.01 | 88.75 ± 2.46 | 364/400 |
| fusion_aug_flip | 0,1,2 | 89.44 ± 0.67 | 90.75 ± 1.15 | 88.83 ± 3.25 | 362/400 |
| tip_adapter | 0,1,2 | 89.17 ± 0.55 | 90.50 ± 1.32 | 88.42 ± 0.14 | 360/400 |
| tip_adapter_f | 0,1,2 | 88.75 ± 0.36 | 90.75 ± 0.75 | 90.00 ± 0.66 | 360/400 |
| linear_head | 0,1,2 | 89.03 ± 0.32 | 90.33 ± 1.66 | 89.08 ± 1.15 | 354/400 |
| transductive | 0,1,2 | 94.65 ± 0.32 | 95.08 ± 0.14 | 90.92 ± 0.29 | 380/400 |
| transductive_sinkhorn | 0,1,2 | 95.56 ± 0.12 | 96.75 ± 0.43 | 90.67 ± 0.58 | 388/400 |

## siglip2_so400m_384

| method | seeds | val (tuned) | test (tuned) | test @ defaults | seed-0 test |
|---|---|---|---|---|---|
| zeroshot | 0,1,2 | 87.50 ± 0.00 | 90.00 ± 0.00 | 90.00 ± 0.00 | 360/400 |
| zeroshot_desc | 0,1,2 | 85.00 ± 0.00 | 88.25 ± 0.00 | 88.25 ± 0.00 | 353/400 |
| zeroshot_both | 0,1,2 | 88.54 ± 0.00 | 90.25 ± 0.00 | 90.25 ± 0.00 | 361/400 |
| proto | 0,1,2 | 73.68 ± 3.30 | 75.42 ± 4.07 | 75.42 ± 4.07 | 283/400 |
| proto_aug | 0,1,2 | 76.60 ± 4.87 | 76.25 ± 3.07 | 76.25 ± 3.07 | 291/400 |
| fusion | 0,1,2 | 90.28 ± 0.32 | 91.92 ± 0.38 | 88.58 ± 3.79 | 366/400 |
| fusion_aug | 0,1,2 | 90.21 ± 0.55 | 91.83 ± 1.01 | 88.25 ± 3.70 | 365/400 |
| fusion_aug_flip | 0,1,2 | 90.28 ± 0.43 | 92.00 ± 0.87 | 88.58 ± 4.22 | 370/400 |
| tip_adapter | 0,1,2 | 90.35 ± 0.12 | 92.33 ± 0.58 | 90.33 ± 0.14 | 368/400 |
| tip_adapter_f | 0,1,2 | 91.11 ± 0.32 | 92.58 ± 0.38 | 91.50 ± 0.43 | 369/400 |
| linear_head | 0,1,2 | 90.35 ± 0.43 | 91.83 ± 1.91 | 91.33 ± 1.42 | 359/400 |
| transductive | 0,1,2 | 93.61 ± 0.12 | 93.83 ± 2.02 | 92.50 ± 0.25 | 380/400 |
| transductive_sinkhorn | 0,1,2 | 94.86 ± 0.12 | 96.67 ± 0.29 | 93.83 ± 0.14 | 386/400 |

## ensemble

| method | seeds | val (tuned) | test (tuned) | test @ defaults | seed-0 test |
|---|---|---|---|---|---|
| tip_adapter_f__all5 | 0,1,2 | 90.49 ± 0.52 | 93.33 ± 0.58 | n/a | 372/400 |
| tip_adapter_f__large4 | 0,1,2 | 90.28 ± 0.67 | 93.00 ± 0.43 | n/a | 370/400 |
| transductive__all5 | 0,1,2 | 94.31 ± 0.73 | 95.42 ± 0.14 | n/a | 381/400 |
| transductive__large4 | 0,1,2 | 94.31 ± 0.24 | 95.08 ± 0.14 | n/a | 380/400 |
| transductive_sinkhorn__all5 | 0,1,2 | 95.21 ± 0.36 | 97.08 ± 0.14 | n/a | 388/400 |
| transductive_sinkhorn__large4 | 0,1,2 | 95.28 ± 0.48 | 97.25 ± 0.00 | n/a | 389/400 |

## Shot sensitivity (20 random shot draws, fixed hyperparameters, no validation labels)

| encoder | method | val | test | test range |
|---|---|---|---|---|
| clip_b16 | proto | 66.05 ± 6.31 | 65.94 ± 7.46 | 49.25 to 77.25 |
| clip_b16 | tip_adapter (alpha 1, beta 5.5) | 85.76 ± 0.27 | 87.50 ± 0.41 | 86.75 to 88.25 |
| clip_l14 | proto | 68.61 ± 6.84 | 68.80 ± 7.33 | 50.25 to 77.50 |
| clip_l14 | tip_adapter (alpha 1, beta 5.5) | 86.33 ± 0.21 | 89.22 ± 0.33 | 88.75 to 90.00 |
| siglip_so400m_224 | proto | 72.91 ± 4.09 | 72.99 ± 4.67 | 65.50 to 84.50 |
| siglip_so400m_224 | tip_adapter (alpha 1, beta 5.5) | 88.13 ± 0.00 | 90.42 ± 0.16 | 90.25 to 90.75 |
| siglip_so400m_384 | proto | 73.50 ± 4.54 | 73.45 ± 4.65 | 67.00 to 81.00 |
| siglip_so400m_384 | tip_adapter (alpha 1, beta 5.5) | 86.35 ± 0.19 | 88.50 ± 0.20 | 88.25 to 88.75 |
| siglip2_so400m_384 | proto | 71.77 ± 4.12 | 72.75 ± 4.47 | 64.00 to 78.75 |
| siglip2_so400m_384 | tip_adapter (alpha 1, beta 5.5) | 88.90 ± 0.17 | 90.41 ± 0.19 | 89.75 to 90.50 |

## Grayscale ablation

Only the Flower class is stored in colour (all 175 Flower images, train and test); the other 15 classes are grayscale files. Here every image is converted to 3-channel grayscale before the encoder, so colour cannot act as a shortcut for Flower. Same shots, grids and selection rule; mean ± std over seeds 0, 1, 2.

| encoder | method | val, original | val, grayscale | test, original | test, grayscale |
|---|---|---|---|---|---|
| clip_b16 | zeroshot_both | 85.42 ± 0.00 | 85.42 ± 0.00 | 87.00 ± 0.00 | 87.00 ± 0.00 |
| clip_b16 | tip_adapter_f | 87.71 ± 0.75 | 87.71 ± 0.75 | 89.58 ± 1.81 | 89.58 ± 1.81 |
| clip_b16 | transductive | 92.78 ± 0.32 | 92.78 ± 0.32 | 92.00 ± 0.43 | 92.00 ± 0.43 |
| clip_b16 | transductive_sinkhorn | 94.10 ± 0.24 | 94.10 ± 0.24 | 94.42 ± 0.14 | 94.42 ± 0.14 |
| clip_l14 | zeroshot_both | 86.25 ± 0.00 | 86.25 ± 0.00 | 88.75 ± 0.00 | 88.75 ± 0.00 |
| clip_l14 | tip_adapter_f | 88.82 ± 0.94 | 88.82 ± 0.94 | 91.92 ± 1.04 | 91.92 ± 1.04 |
| clip_l14 | transductive | 92.99 ± 0.52 | 92.99 ± 0.52 | 95.75 ± 0.00 | 95.75 ± 0.00 |
| clip_l14 | transductive_sinkhorn | 95.07 ± 0.24 | 95.07 ± 0.24 | 96.25 ± 0.00 | 96.25 ± 0.00 |
| siglip_so400m_224 | zeroshot_both | 88.13 ± 0.00 | 88.13 ± 0.00 | 90.25 ± 0.00 | 90.25 ± 0.00 |
| siglip_so400m_224 | tip_adapter_f | 90.28 ± 1.15 | 90.14 ± 1.39 | 92.00 ± 0.43 | 92.00 ± 0.43 |
| siglip_so400m_224 | transductive | 94.72 ± 0.24 | 94.72 ± 0.24 | 94.92 ± 0.29 | 94.92 ± 0.29 |
| siglip_so400m_224 | transductive_sinkhorn | 95.56 ± 0.32 | 95.56 ± 0.32 | 97.00 ± 0.43 | 97.00 ± 0.43 |
| siglip_so400m_384 | zeroshot_both | 86.25 ± 0.00 | 86.25 ± 0.00 | 88.25 ± 0.00 | 88.25 ± 0.00 |
| siglip_so400m_384 | tip_adapter_f | 88.75 ± 0.36 | 88.75 ± 0.36 | 90.75 ± 0.75 | 90.83 ± 0.63 |
| siglip_so400m_384 | transductive | 94.65 ± 0.32 | 94.72 ± 0.24 | 95.08 ± 0.14 | 95.08 ± 0.14 |
| siglip_so400m_384 | transductive_sinkhorn | 95.56 ± 0.12 | 95.56 ± 0.12 | 96.75 ± 0.43 | 96.75 ± 0.43 |
| siglip2_so400m_384 | zeroshot_both | 88.54 ± 0.00 | 88.54 ± 0.00 | 90.25 ± 0.00 | 90.25 ± 0.00 |
| siglip2_so400m_384 | tip_adapter_f | 91.11 ± 0.32 | 91.11 ± 0.32 | 92.58 ± 0.38 | 92.58 ± 0.38 |
| siglip2_so400m_384 | transductive | 93.61 ± 0.12 | 93.61 ± 0.12 | 93.83 ± 2.02 | 93.83 ± 2.02 |
| siglip2_so400m_384 | transductive_sinkhorn | 94.86 ± 0.12 | 94.86 ± 0.12 | 96.67 ± 0.29 | 96.67 ± 0.29 |

Reading: removing colour changes the Flower embeddings (cosine to the colour version about 0.87 to 0.90) but not the predictions; accuracy is unchanged to within one image. Colour is not what makes Flower easy, and the other 15 classes never had colour.

## Does the one shot matter?

`none` = no training image (prompts only); `random` = the three seeded shots (mean ± std); `typical` / `atypical` = ORACLE choices, the training image closest to / farthest from its class mean embedding. The oracle rows need all training labels and are analysis only, never a reported one-shot result. Raw shot embedding (no augmented views), so `random` differs slightly from the main tables. Hyperparameters tuned on val as everywhere else; `test @ defaults` uses no validation label. See `scripts/shot_matters.py`.

| encoder | shot | method | val (tuned) | test (tuned) | test @ defaults |
|---|---|---|---|---|---|
| clip_b16 | none | transductive | 92.92 | 91.75 | 91.75 |
| clip_b16 | none | transductive_sinkhorn | 93.75 | 94.50 | 92.00 |
| clip_b16 | random | tip_adapter | 87.22 ± 0.52 | 88.67 ± 2.13 | 87.42 ± 0.14 |
| clip_b16 | random | transductive | 92.57 ± 0.84 | 92.83 ± 0.88 | 92.08 ± 0.52 |
| clip_b16 | random | transductive_sinkhorn | 94.17 ± 0.36 | 94.75 ± 0.43 | 91.92 ± 0.52 |
| clip_b16 | typical | tip_adapter | 91.67 | 91.50 | 87.25 |
| clip_b16 | typical | transductive | 93.12 | 94.25 | 92.25 |
| clip_b16 | typical | transductive_sinkhorn | 93.96 | 94.50 | 92.25 |
| clip_b16 | atypical | tip_adapter | 86.04 | 87.50 | 87.25 |
| clip_b16 | atypical | transductive | 91.46 | 93.00 | 88.75 |
| clip_b16 | atypical | transductive_sinkhorn | 93.75 | 93.75 | 92.25 |
| clip_l14 | none | transductive | 93.75 | 93.50 | 94.00 |
| clip_l14 | none | transductive_sinkhorn | 95.21 | 96.25 | 94.25 |
| clip_l14 | random | tip_adapter | 87.08 ± 0.95 | 90.83 ± 1.81 | 89.17 ± 0.38 |
| clip_l14 | random | transductive | 92.92 ± 0.83 | 95.08 ± 0.95 | 92.92 ± 0.88 |
| clip_l14 | random | transductive_sinkhorn | 95.00 ± 0.21 | 96.25 ± 0.00 | 94.42 ± 0.14 |
| clip_l14 | typical | tip_adapter | 90.62 | 92.50 | 89.50 |
| clip_l14 | typical | transductive | 93.75 | 93.50 | 93.50 |
| clip_l14 | typical | transductive_sinkhorn | 95.00 | 96.25 | 94.00 |
| clip_l14 | atypical | tip_adapter | 86.46 | 88.75 | 88.50 |
| clip_l14 | atypical | transductive | 93.12 | 95.75 | 91.00 |
| clip_l14 | atypical | transductive_sinkhorn | 95.21 | 96.25 | 94.50 |
| siglip_so400m_224 | none | transductive | 95.00 | 94.50 | 93.25 |
| siglip_so400m_224 | none | transductive_sinkhorn | 95.00 | 94.50 | 92.75 |
| siglip_so400m_224 | random | tip_adapter | 89.86 ± 0.32 | 92.00 ± 1.15 | 90.50 ± 0.00 |
| siglip_so400m_224 | random | transductive | 94.65 ± 0.12 | 94.75 ± 0.00 | 92.67 ± 0.63 |
| siglip_so400m_224 | random | transductive_sinkhorn | 95.28 ± 0.12 | 96.83 ± 0.38 | 93.17 ± 0.14 |
| siglip_so400m_224 | typical | tip_adapter | 92.50 | 93.00 | 90.75 |
| siglip_so400m_224 | typical | transductive | 95.63 | 95.75 | 93.25 |
| siglip_so400m_224 | typical | transductive_sinkhorn | 95.21 | 89.75 | 93.00 |
| siglip_so400m_224 | atypical | tip_adapter | 88.13 | 90.25 | 90.25 |
| siglip_so400m_224 | atypical | transductive | 94.58 | 95.00 | 91.75 |
| siglip_so400m_224 | atypical | transductive_sinkhorn | 95.00 | 96.75 | 93.25 |
| siglip_so400m_384 | none | transductive | 94.58 | 87.00 | 89.50 |
| siglip_so400m_384 | none | transductive_sinkhorn | 95.00 | 88.75 | 89.50 |
| siglip_so400m_384 | random | tip_adapter | 89.17 ± 0.55 | 90.50 ± 1.32 | 88.42 ± 0.14 |
| siglip_so400m_384 | random | transductive | 94.58 ± 0.00 | 95.08 ± 0.14 | 91.17 ± 0.52 |
| siglip_so400m_384 | random | transductive_sinkhorn | 95.42 ± 0.00 | 96.58 ± 0.52 | 90.58 ± 0.76 |
| siglip_so400m_384 | typical | tip_adapter | 92.71 | 91.25 | 88.50 |
| siglip_so400m_384 | typical | transductive | 95.00 | 88.75 | 91.25 |
| siglip_so400m_384 | typical | transductive_sinkhorn | 95.42 | 97.50 | 90.25 |
| siglip_so400m_384 | atypical | tip_adapter | 86.46 | 88.25 | 88.25 |
| siglip_so400m_384 | atypical | transductive | 92.08 | 89.50 | 90.75 |
| siglip_so400m_384 | atypical | transductive_sinkhorn | 93.54 | 89.00 | 89.75 |
| siglip2_so400m_384 | none | transductive | 94.79 | 95.25 | 92.25 |
| siglip2_so400m_384 | none | transductive_sinkhorn | 94.79 | 96.00 | 93.75 |
| siglip2_so400m_384 | random | tip_adapter | 90.35 ± 0.12 | 92.33 ± 0.58 | 90.33 ± 0.14 |
| siglip2_so400m_384 | random | transductive | 93.54 ± 0.00 | 95.00 ± 0.00 | 92.42 ± 0.38 |
| siglip2_so400m_384 | random | transductive_sinkhorn | 94.79 ± 0.00 | 96.75 ± 0.25 | 93.83 ± 0.29 |
| siglip2_so400m_384 | typical | tip_adapter | 93.54 | 94.50 | 90.50 |
| siglip2_so400m_384 | typical | transductive | 94.79 | 92.75 | 92.00 |
| siglip2_so400m_384 | typical | transductive_sinkhorn | 95.00 | 96.75 | 93.75 |
| siglip2_so400m_384 | atypical | tip_adapter | 88.75 | 90.25 | 90.00 |
| siglip2_so400m_384 | atypical | transductive | 93.54 | 94.75 | 90.25 |
| siglip2_so400m_384 | atypical | transductive_sinkhorn | 94.79 | 96.25 | 93.75 |

Reading: for the inductive Tip-Adapter the shot matters (typical shots beat random ones, atypical ones are worse). For transductive refinement, validation accuracy is about the same with no shot at all as with a random shot: the unlabelled queries and the prompts do the work. The single-run test numbers without a shot or with an oracle shot are less stable: in a few cases the setting chosen on validation merges two similar classes on the 400-image test batch and test accuracy drops to 87 to 90 %, which the three random-shot seeds never did.
