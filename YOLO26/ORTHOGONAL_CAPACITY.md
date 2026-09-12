# Orthogonal-Capacity: accuracy-first revision

The user's revised priority is to retain learning capacity while remaining lighter than **stock YOLO11**, not to minimize parameters relative to Orthogonal itself. This supersedes the earlier recommendation of B+C ratio0.25. Lower parameter count does not by itself establish accuracy loss or gain; neither does adding capacity establish better accuracy.

## Architecture choice

Start from the original Orthogonal architecture. Restore all full-width dense/OHR spatial operators in downsamplers, regression towers and CSP blocks. No CenterRankConv, no OrthoRankDetect and no selected-decoding or Efficient-fusion changes are included. Increase existing CSP hidden widths in four locations:

| Location | Original expansion | Capacity expansion | Hidden channels at n |
|---|---:|---:|---:|
| P5 backbone, node8 | 0.25 | 0.375 | 64 ->96 |
| P4 top-down refinement, node12 | 0.25 | 0.375 | 32 ->48 |
| P3 output refinement, node14 | 0.375 | 0.5 | 24 ->32 |
| P4 bottom-up refinement, node17 | 0.25 | 0.375 | 32 ->48 |

The allocation strengthens coarse semantic processing, intermediate cross-scale refinement and the high-resolution detection path without constraining neighboring spatial interactions to a low-rank subspace. Output widths/strides, original OSIFusion, classifier, regression widths, reg_max16 and end-to-end assignment remain unchanged from Orthogonal. This is a configuration-level capacity allocation hypothesis using existing modules, not a scientific originality claim. Small-object or mAP benefits are **CHƯA ĐƯỢC XÁC MINH BẰNG THỰC NGHIỆM.**

Selected model: `ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal-capacity.yaml`. The original YAML and old rank YAMLs remain available. The original root `train.py` is unchanged; use `train_orthogonal_capacity.py`. The shared `train_orthogonal_rank.py` runner now also defaults to `--variant capacity`; old A/B/C variants require explicit selection. `--variant yolo11` means the actual stock YOLO11 baseline; the historical `--variant baseline` flag still means original Orthogonal for backward compatibility.

## Actual parameter budgets

All counts come from constructing and fusing this repository's implementations. Stock YOLO11 has a one-to-many head; Orthogonal has two training heads and removes one during deployment. Thus training-form and deployment counts are reported separately. Their assignment/training behavior differs and counts alone do not compare accuracy.

| Scale, nc80 | Stock training params | Capacity training params | Stock deploy params | Capacity deploy params |
|---|---:|---:|---:|---:|
| n | 2,624,080 | 2,587,288 | 2,616,248 | 1,959,072 |
| s | 9,458,752 | 8,305,272 | 9,443,760 | 6,820,728 |
| m | 20,114,688 | 17,726,904 | 20,091,712 | 15,436,744 |
| l | 25,372,160 | 17,726,904 | 25,340,992 | 15,436,744 |
| x | 56,966,176 | 39,702,744 | 56,919,424 | 34,647,240 |

The local Orthogonal YAML uses one outer repeat at these stages, so its m/l versions have the same parameter count even though stock YOLO11's m/l differ. This was not changed. At nc4, stock YOLO11n has2,590,620 training/2,582,932 deploy parameters; Capacity has2,520,368/1,925,756. Budget checks apply to these measured configurations; other class counts/settings need checking again.

## Comparable CPU timing endpoint

`research/validate_orthogonal_capacity.py` measures the same endpoint for all candidates: fused backbone + neck + dense raw box/class logits. DFL/decode/top-k/NMS and preprocessing/IO are excluded from all models. Stock one-to-many logits and Orthogonal fused one-to-one logits share shapes but are not interchangeable accuracy outputs. To expose the common endpoint, only the head's return-mode flag is set to training; all BN children remain in eval and no gradients are enabled. These numbers must not be compared directly with older timings that included decoding/top-k.

CPU, torch2.2.2+cpu, one thread, FP32, batch1,640 square,nc80;15 warmups/model,40 randomized paired samples:

| Model | Raw Conv-only GFLOPs | Mean ms | Median ms | P95 ms | FPS |
|---|---:|---:|---:|---:|---:|
| Stock YOLO11n | 6.4799872 | 129.892 | 128.640 | 133.406 | 7.699 |
| Original Orthogonal | 5.5632000 | 123.196 | 121.793 | 129.564 | 8.117 |
| Previous rank B+C | 3.7798016 | 112.528 | 111.491 | 118.911 | 8.887 |
| **Orthogonal-Capacity** | **5.8744960** | **126.542** | **126.019** | **131.762** | **7.902** |

Capacity intentionally spends more computation than the compact candidate. This run still puts it below stock at the measured common endpoint. Conv-only counts exclude stock attention matmuls, all elementwise work and decoding; they are not total theoretical FLOPs. CPU measurements do not establish GPU/end-to-end speed or peak-memory behavior. Evidence and raw timing samples are in `experiments/orthogonal_capacity/validation_nc80.json` and the separate nc4 run. Within-run bootstrap intervals do not cover training variance or uncontrolled hardware state.

The separate nc4 run was highly variable: stock mean200.388/median126.896/P95379.799 ms; Capacity204.380/125.612/395.199 ms. Capacity was **not faster in mean** in that run. Keep those samples and do not use the nc80 result alone to claim reliable acceleration. The structural parameter/Conv budget checks still pass at nc4 (Capacity5.7072128 versus stock6.312704 raw Conv-GFLOPs). This revision prioritizes capacity within a lighter model budget, not a guaranteed latency win on this uncontrolled CPU session.

## Training and decision

```powershell
# Prepare the exact stock/candidate recipes without training:
.venv/Scripts/python.exe train_orthogonal_capacity.py --variant yolo11 --prepare-only --seeds 0 1 2
.venv/Scripts/python.exe train_orthogonal_capacity.py --prepare-only --seeds 0 1 2

# Full training requires an appropriate environment; current torch is CPU-only:
.venv/Scripts/python.exe train_orthogonal_capacity.py --variant yolo11 --device 0 --amp --seeds 0 1 2
.venv/Scripts/python.exe train_orthogonal_capacity.py --device 0 --amp --seeds 0 1 2
```

For CPU runs use `--device cpu`; `--no-plots` bypasses the absent optional polars plotting dependency while retaining metrics/CSV. Existing original Orthogonal weights can initialize matching layers via --init, but wider tensors are newly initialized and explicitly listed. Class order/scale/top-level architecture must match. This is not a strict transfer from original Orthogonal. Do not use a stock YOLO11 checkpoint as though it were an Orthogonal checkpoint.

Decision: replace the previous efficiency-first recommendation with this capacity-first **training candidate**. Accuracy validation has priority over further compression. Full-dataset baseline reproduction, mAP50/mAP50-95, APsmall and seed variance are still required before accepting an accuracy/complexity Pareto claim. No claim of superior accuracy, best-on-all-data performance or publication-level originality is made.

Validation completed: real detection loss/backward and eval-to-fused raw-logit checks at nc4/80; all-scale parameter construction checks; official dynamic ONNX export and ORT CPU raw-logit comparison; actual YOLO checkpoint save/load during a one-epoch trainer run. The trainer run uses the prior15-image/7-validation-image smoke subset with4 classes and partial initialization from its original-Orthogonal checkpoint. It completed with mAP50/mAP50-95=0; these small-run metrics validate software integration only, not learning quality. Artifacts and initialization details are in `experiments/orthogonal_capacity/trainer_smoke/`. Full-dataset training recipes are prepared separately in `experiments/orthogonal_capacity/training/`.
