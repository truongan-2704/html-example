# YOLO11-Orthogonal: improvement audit and implementation rationale

Update 2026-09-12: a subsequent selected-decoding ablation is documented in `ORTHOGONAL_DECODE_STUDY.md`. Its isolated decode cost decreased, but whole-model incremental acceleration over Efficient was not established; the combined configuration remains experimental. Later CPU timing runs also show substantial session variability, so the timing table below is evidence for its original run only. The new nearest-work audit is `NOVELTY_MATRIX.md`.

Date: 2026-09-11. Scope: improve the existing Orthogonal implementation without claiming universal superiority. Original YAML remains available. No evidence currently supports best-on-all-datasets or paper-level novelty.

## Audit before implementation

Actual YAML: `ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal.yaml`, scale n, nc=80, end2end=True, reg_max default 16. Backbone has C3k2/C3k2_Ortho, SPPF and SimAM; neck has four OSIFusion nodes plus bottom-up convolutions/refinement; Detect takes nodes 14/17/20. SESPGate and C2PSA_Ortho are not used by this YAML. Containers still concatenate internally. Static positive normalized OSIFusion weights are BiFPN-style, not input-dependent spectral interactions.

Measured before editing: 2,383,960 training-form parameters, 1,813,504 after deployment fusion/head removal, random-input 1×3×128×160 produces 1×300×6 output. Snapshot with config, weights, input and raw output saved to `experiments/orthogonal_audit/before_changes.pt`. These are not trained detector results. Prior docs/YAML contain different architectural revisions and numerical claims without linked reproducible evidence; they are not accepted as benchmarks.

Reproduced correctness defect: OHRConv(8,8).double().eval(), forward, fuse, forward fails because newly constructed fused_conv defaults to float32. The same allocation site defaults to CPU regardless of source device. Fixing that allocation is required for deployment portability; CPU FP32 forward/default YOLO11 topology need not change. Patch only allocation dtype/device and child mode, and test nontrivial BN statistics and multiple groups/strides.

## Research hypotheses and decisions

All accuracy/performance expectations without measurements: **CHƯA ĐƯỢC XÁC MINH BẰNG THỰC NGHIỆM.**

| Hypothesis | Mechanism / anticipated benefit | Drawback / complexity / novelty risk | Quantitative expectation, not a measured accuracy claim |
|---|---|---|---|
| H1: projection after upsample wastes neck compute | In eval, commute pointwise projection with nearest resize | Training BN statistics do not generally commute on odd shapes; low implementation complexity, HIGH prior-art risk | For 2×2 enlargement, projection MACs decrease 75% at that node, parameters unchanged; mathematical equivalent eval function |
| H2: deployment conversion must preserve tensor device/dtype | Allocate fused kernel on source tensor device and dtype | Small floating rounding differences in low precision; low complexity; bug fix, not research novelty | Params/FLOPs unchanged; avoids reproduced FP64 failure, CUDA test conditional |
| H3: dense P3 regression can dominate deployment budget | Explore low-rank/partial spatial box branch with same outputs | May hurt localization/APsmall; medium complexity; HIGH overlap with lightweight heads | Rank-r 3×3 then 1×1 costs 9Ci·r+r·Co versus 9Ci·Co per pixel; no whole-model reduction claim yet |
| H4: normalized ReLU scale weights can shut off a branch | Explore constrained positive routing with measured scale utilization | Changing gate changes training and possibly accuracy; HIGH BiFPN/ASFF overlap | Scalar parameter count essentially unchanged; FLOPs reduction not expected |
| H5: rich train-time branches may be redundant | Compare removing directional or identity branches, preserving deploy kernel size | Potential optimization/regularization loss; low complexity; HIGH ACNet/RepVGG overlap | Removing 1×3+3×1 saves 6Ci·Co/groups kernel weights before fusion, no deployment FLOP saving by itself |

Selection for this change: H1 engineering extension and H2 correctness fix. H3–H5 are deferred until trained baseline/ablation budget exists. This is not a final new detector claimed to improve mAP.

## Nearest-work / novelty matrix

| Idea | Closest source | Similarity / difference | Assessment |
|---|---|---|---|
| Orthogonal directional branches | [ACNet, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.html) | 3×3 + asymmetric branches and deploy sum; extra 1×1/identity resembles rep blocks | HIGH overlap; no independent orthogonality constraint in current code |
| 1×1/identity BN reparameterization | [RepVGG](https://arxiv.org/abs/2101.03697), [DBB](https://arxiv.org/abs/2103.13425) | Existing linear branch fusion algebra | Established mechanism, not independent contribution |
| Positive normalized scale fusion | [EfficientDet/BiFPN](https://arxiv.org/abs/1911.09070) | Current OSIFusion uses two ReLU-normalized scalar weights | HIGH overlap; renaming is insufficient |
| Projection before resolution enlargement | [FPN](https://arxiv.org/abs/1612.03144) and pointwise operator commutation | Common low-resolution projection principle; this change specifically preserves old training path and checkpoint keys | Engineering optimization; not claiming publication novelty |
| Device/dtype-preserving fuse | PyTorch tensor/module contract | Portability bug fix | No scientific novelty claim |

No fabricated percentage of conceptual similarity is assigned. Strong similarity is sufficient to withhold novelty claims here.

## Mathematical formulation of selected extension

Let U be nearest-neighbor resizing (selection/replication of spatial positions), P a 1×1 Conv + frozen BN + elementwise SiLU, L local feature and G coarse feature. Current fusion is Y = SiLU(a·P_local(L) + b·P_global(U(G))). In eval, P acts independently with identical parameters at each pixel, hence P(U(G)) = U(P(G)) in real arithmetic. Optimized eval is Y = SiLU(a·P_local(L) + b·U(P_global(G))). It supports noninteger/odd target sizes because nearest U still selects pixels.

Training is deliberately unchanged: BatchNorm uses batch/spatial statistics and arbitrary nearest resize can change their weighting. The new forward delegates to the original path when training. When the source is larger than target, retain resize-first to avoid doing more projection computation. No spatial convolution or interpolation with mixing is commuted.

For G of size B×Cg×Hg×Wg and projected channels Co, projection MACs change from B·Ht·Wt·Cg·Co to B·Hg·Wg·Cg·Co for enlargement. Kernel/BN parameters are unchanged. Upsampled tensor channels become Co instead of Cg; when Co<Cg its storage decreases by B·Ht·Wt·(Cg−Co) elements, but this is not peak-memory or bandwidth measurement. New low-resolution projected temporary exists. Actual latency must be measured.

## Extension and compatibility contract

Add OSIFusionEfficient in a separate file and a separate `yolo11-orthogonal-efficient.yaml`; same parameters and state_dict keys as OSIFusion, strict weight transfer. Minimal module export/parser registration lets normal YOLO API parse the new YAML. Baseline YAML keeps old OSIFusion. Existing fuse dispatch recognizes the subclass through OSIFusion. The small OHR allocation fix applies to old and new Orthogonal; does not affect stock YOLO11 modules. Tests must check stock snapshot, old Orthogonal snapshot, trained BN fuse, raw outputs, gradients, AMP, shapes, save/load and export as available.

No new training will start until tests pass. No mAP/FPS improvement claim will be made without matching evidence; synthetic model-only timing is not accuracy reproduction or end-to-end performance.

## Additional reproduced defect and minimal patch

SimAM returned NaN on B×C×1×1 input because its unbiased spatial-variance denominator was zero. Replaced n=HW−1 with max(HW−1,1). Centered singleton variance is zero; output is sigmoid(0.5)·X with finite gradients. This does not change the computation for HW>1. Full Orthogonal eval on 2×3×32×32 is now finite. This patch is not a new attention mechanism and has no parameter cost.

Tracked-source patch surface is deliberately small: OHR fused allocation dtype/device/mode; SimAM singleton denominator; one new module export and parser registration. Original Orthogonal and YOLO11 YAML files are unchanged. Standard model forward snapshots still match exactly. The old design document remains historical; claims in it such as guaranteed orthogonality, zero runtime, fixed percentage memory savings and universal superiority are not endorsed by this implementation.

## Measured results, CPU only

Evidence: `experiments/orthogonal_audit/benchmark.json`. 13th Gen Intel Core i7-13650HX (name reported by export utility), Windows, torch 2.2.2+cpu, FP32, one CPU thread, eager fused model, batch 1, 640×640, same random weights and input. Twenty warmups per model, sixty randomized alternating pairs. Timing includes built-in end-to-end head top-k; excludes preprocessing, IO and rendering. No accuracy training was performed.

| Model | Deploy parameters | Conv-only GFLOPs (2/MAC) | Mean ms | Median ms | P95 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Orthogonal original | 1,813,504 | 5.5642752 | 131.078 | 130.625 | 135.198 | 7.629 |
| Orthogonal Efficient | 1,813,504 | 5.4069888 | 127.369 | 125.979 | 131.220 | 7.851 |

Conv count decreases 0.1572864 GFLOPs (about 2.83%). This count excludes SimAM, activation, resize, normalization, decode and top-k arithmetic: it is not the total of all operations. Mean latency decreases about 2.83% in this run. Paired original-minus-efficient mean difference is 3.708 ms; within-run bootstrap 95% interval [1.806,5.288] ms. This interval does not quantify hardware variability across runs or training seed variance. It does not support a universal hardware speed claim. GPU memory, GPU latency, end-to-end image pipeline latency and mAP/APsmall remain NOT_EVALUATED.

## Validation and limitations

Eleven unittest methods: ten pass, CUDA test skipped. Subcases cover float32/float64, groups/depthwise/stride, nontrivial BN statistics, repeated fuse, odd/even and mixed-resolution fusion, input preservation, exact training output/gradients/BN state, CPU BF16 autocast forward/backward, real model end-to-end loss/backward, all n/s/m/l/x scales, strict state_dict transfer, YOLO API checkpoint roundtrip, singleton features and both historical local forward snapshots. API save uses FP16 weights, so serialized API checkpoint comparisons use an explicit 2e-3 tolerance, separately from exact state_dict snapshot checks.

ONNX 1.14.0, opset17, ONNX Runtime1.15.1 CPU: official YOLO export with dynamic batch/H/W passes graph checking and finite output checks for 1×3×128×160 and 2×3×160×192 (max_det=100). Raw one-to-one box/class logits exported separately agree with PyTorch at rtol=atol=1e-4; observed max absolute error 2.3842e-7. Post-topk class/box ordering is not asserted equal because untrained logits have ties. Exported dimensions must provide at least max_det anchor locations, as required by the existing export head; the eager singleton fix does not remove that export constraint. TensorRT, CUDA, INT8 and trained-checkpoint AP equivalence have not been tested.

## Usage and reproducibility

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal-efficient.yaml')
# Load your existing Orthogonal checkpoint via model.load(checkpoint_path).
# Same channel/config/scale is required for full weight compatibility.
```

Select the original `yolo11-orthogonal.yaml` to retain the old fusion scheduling. `train.py` remains unchanged so existing recipe selection does not silently switch. Training the new YAML uses the original OSIFusion computation; the optimized path activates in eval. Never use `efficient_untrained.pt` for real detections: it is an export test artifact with random weights.

```powershell
.venv/Scripts/python.exe -B research/test_orthogonal_efficient.py
.venv/Scripts/python.exe -B research/benchmark_orthogonal.py
.venv/Scripts/python.exe -B research/export_orthogonal_probe.py
```

The benchmark/probe use the preserved pre-edit snapshot `experiments/orthogonal_audit/before_changes.pt`. Keep it unchanged for regression evidence. Full research decision remains pending accuracy reproduction. Accept the two correctness fixes and opt-in execution optimization as engineering changes; do not call the result a paper-level architecture contribution. Next experiment must establish a trained Orthogonal baseline, then investigate H3–H5 with controlled ablations and external dataset validation.
