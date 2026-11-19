# $\color{#800080}{PLUM}$ (Pooled, Leveled, Unit-norm Mahalanobis)

| Paper | Code |
|---|---|
| Box feature: res5 RoIAlign 7x7 mean at the box the detector reads (proposal for Faster R-CNN, output box for Deformable DETR) | `plum/frcnn.py`, `plum/ddetr.py` |
| Unit norm, class means, shared Ledoit-Wolf covariance, $-\min_c$ Mahalanobis | `plum/score.py` |
| Split of the ID training images (even / %4==1 / %4==3) | `plum/method.py` |
| Strata (FPN level x border contact), line in log area, rank | `plum/level.py` |
| AUROC / FPR95, ID positive | `plum/metrics.py` |

Faster R-CNN follows the VOS protocol: a detection is kept iff the maximum foreground softmax of its proposal reaches the
F1-optimal threshold on ID val. Deformable DETR follows the SIREN protocol: every last-layer (query, class) pair with
sigmoid > 0.1. The ID training images run through the same rule. VOC uses all 16,551 training images, BDD the 10,000
images listed in `plum/bdd_train_10k.txt`.

## Setup

- Python 3.12 and the packages of `requirements.txt` (PyTorch 2.14, CUDA 13.2 wheels).
- detectron2 0.6 and the deformable-attention op of SIREN, both built from source. SIREN needs the two PyTorch 2 patches
  in `patches/`.
- `PLUM_DATA_ROOT`: the VOS / SIREN benchmark datasets and the public vanilla checkpoints of VOS (Faster R-CNN R50-FPN)
  and SIREN (Deformable DETR). Dumps are written to `$PLUM_DATA_ROOT/dumps/`.
- `PLUM_THIRD_PARTY`: the VOS and SIREN code (`vos/`, `siren/detr/`).
- Once, for BDD: `python scripts/fix_bdd_train_ids.py`. The VOS BDD train json gives 10 pairs of images the same id; the
  script gives the first image of each pair a new id and lists these images, which PLUM leaves out.

## Run

From this folder:

```bash
python -m run.frcnn_threshold --task voc      # VOS threshold, expected 0.5728 (BDD 0.4006)
python -m run.frcnn_threshold --task bdd
python -m run.dump_frcnn --task voc           # ID val, OOD sets, ID training images
python -m run.dump_frcnn --task bdd
python -m run.dump_ddetr --task voc
python -m run.dump_ddetr --task bdd
python -m run.evaluate                        # AUROC / FPR95 of all four detector x task pairs
```

Inference runs in FP32 with TF32 disabled.

## Expected output

| Detector | ID | MS-COCO AUROC / FPR95 | OpenImages AUROC / FPR95 |
|---|---|---|---|
| Faster R-CNN R50 | PASCAL-VOC | 91.46 / 44.16 | 94.50 / 29.25 |
| Faster R-CNN R50 | BDD100K | 95.07 / 24.35 | 97.51 / 13.87 |
| Deformable DETR | PASCAL-VOC | 90.74 / 43.17 | 95.07 / 25.59 |
| Deformable DETR | BDD100K | 93.00 / 37.34 | 95.78 / 19.29 |
