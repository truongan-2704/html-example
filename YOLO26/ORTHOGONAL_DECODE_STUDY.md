# Orthogonal selected decoding: hypothesis and evaluation

Date: 2026-09-12. Current decision: REVISE for deployment selection; the implementation passes numerical checks but whole-model speed benefit over Efficient is not established. The hypothesis below was recorded before implementation.

## Audited bottleneck and hypothesis H6

Local `Detect.forward` calls `_inference` (DFL and coordinate decoding at every position), then `postprocess` selects detections using class scores alone. At 640 x 640 the three strides provide N=8400 positions; max_det defaults to K=300. Regression convolutions still need all positions. Hypothesis: gathering the selected distributions before DFL can reduce decoding work and temporary tensors while preserving the existing selection rule and training computation. Actual model latency may improve little or worsen because gather also costs time.

Let Z in R^(B x 4R x N) be regression logits and S in R^(B x C x N) class logits. Obtain score, class and position indices I using the **unchanged** two-stage `Detect.get_topk_index(sigmoid(S).transpose(1,2), K)`. DFL and distance decoding act independently at each position, hence gather_I(decode(DFL(Z), A, stride)) = decode(DFL(gather_I(Z)), gather_I(A), gather_I(stride)). Repeated positions for different classes must remain repeated. Sigmoid remains before top-k to retain floating-point tie behavior, including saturation.

DFL arithmetic and its probability intermediate change from O(B*4*R*N) to O(B*4*R*K); parameters and regression/classification convolution work do not change. Anchor storage remains O(N), regression logits remain O(B*4*R*N), and score selection retains its original cost. This is not a claim of an N/K whole-model speedup or peak-memory reduction. Training and non-end-to-end inference must delegate to the original head.

## Closest implementations and novelty assessment

| Candidate | Verified closest implementation | Similarity | Difference | Decision |
|---|---|---|---|---|
| Select distributions before DFL/decode | [D-Robotics YOLO conversion guide](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/samples/vision/ultralytics_yolo/conversion/README.md), accessed 2026-09-12 | Explicitly gathers box distributions after candidate filtering, then computes DFL and decodes | This integration preserves local end-to-end two-stage class-aware top-k, training state and checkpoint keys | Same central idea; reject a scientific originality claim |
| Candidate selection before box decode | [MMDetection BaseDenseHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/base_dense_head.py), accessed 2026-09-12 | Filters scores before bounding-box decoding | Local DFL placement and end-to-end output contract | Established execution pattern |

This candidate is an opt-in deployment optimization. It does not satisfy the requested paper-level contribution by itself. Accuracy or universal performance improvement: **CHƯA ĐƯỢC XÁC MINH BẰNG THỰC NGHIỆM.**

## Validation plan and decision rule

Use separate head-only and fusion+head YAMLs for a 2 x 2 execution ablation against original and Efficient. Strictly load the same preserved random weights. Test raw synthetic logits (nontrivial values, ties, repeated positions), batch > 1, class-aware/agnostic selection, reg_max 1/16, eager K>N, exact training gradients, full model, checkpoint and historical baseline snapshots. Verify dynamic ONNX output against a numerical oracle, not only output shapes. Benchmark randomized paired order, warmup, mean/median/P95/FPS, same CPU/thread/input/precision. Retain original configurations. Do not promote the combined configuration on theoretical FLOPs alone. CUDA, trained mAP, end-to-end image latency and independent training seeds require separate evidence.

## Implementation and compatibility

`ultralytics/nn/modules/selected_decode.py` adds `SelectDecodeDetect(Detect)`. Only eval with end2end=True uses the selected decoding path. Class-aware/agnostic top-k, duplicate positions, score sigmoid, anchor construction and distance decoding retain the original rules. There are no new weights, no weight conversion and no change to the loss. The original `head.py` is untouched. Minimal integration patch: export the new class and include it in the parser's detection-head/legacy sets. The explicit forward override preserves the unfused raw one-to-many dictionary and fused empty dictionary as well as one-to-one predictions; an upstream Detect contract change requires revisiting this override.

Two added YAMLs in `ultralytics/cfg/models/11/yolo11-Orthogonal/`:

- `yolo11-orthogonal-sd.yaml`: selected decoding only.
- `yolo11-orthogonal-efficient-sd.yaml`: Efficient fusion plus selected decoding.

Both are **experimental ablation configurations**, not a replacement for `yolo11-orthogonal-efficient.yaml` or the original YAML. All four configurations match after normalizing just these module names. Strict state_dict transfer requires the same scale, channels, class count and end2end configuration.

## Measured model-only CPU results

Artifacts: `experiments/orthogonal_selected/benchmark_640_run1.json` and `benchmark_640_run2.json`. Windows, Intel i7-13650HX, torch2.2.2+cpu, one CPU thread, FP32, fused eager, batch1, 640x640, same random weights/input. Each run has 20 warmups per model and 60 samples in randomized four-model order. All models' output tensors agree with the original at rtol=atol=1e-4. These are timing repetitions, **not training seeds**. No concurrent benchmark/export/test process was launched by this workflow during timing; external system activity and power/clock state were not controlled.

| Model | Deploy params | Conv-only GFLOPs | Run 1 mean / median / P95 ms | Run 1 FPS | Run 2 mean / median / P95 ms | Run 2 FPS |
|---|---:|---:|---|---:|---|---:|
| Original Orthogonal | 1,813,504 | 5.5642752 | 186.640 / 183.920 / 221.167 | 5.358 | 157.572 / 156.298 / 173.003 | 6.346 |
| Efficient | 1,813,504 | 5.4069888 | 174.273 / 170.406 / 209.267 | 5.738 | 153.309 / 150.741 / 174.451 | 6.523 |
| Selected decoding | 1,813,504 | 5.5632384 | 180.866 / 179.985 / 205.932 | 5.529 | 156.199 / 154.424 / 181.737 | 6.402 |
| Efficient + selected | 1,813,504 | 5.4059520 | 176.654 / 173.591 / 219.090 | 5.661 | 154.113 / 151.948 / 176.894 | 6.489 |

Conv counts use 2 FLOPs/MAC and include DFL's fixed projection convolution; they exclude softmax/sigmoid, resize, top-k, gather, SimAM and other elementwise work. DFL hook inputs are actually [1,64,8400] versus [1,64,300]. A small Conv-GFLOP reduction does not describe the full DFL saving.

Paired mean savings of Efficient minus Efficient+selected are **-2.381 ms** (bootstrap95% [-9.068,3.588]) and **-0.804 ms** ([-5.731,3.807]). The combined model was slightly slower on average in both runs, with uncertainty spanning zero. The earlier 131/127 ms measurement in `ORTHOGONAL_IMPROVEMENT.md` belongs to a different session; never subtract those times from these to assert a code effect. Absolute timings vary substantially across runs. Bootstrap intervals describe within-run timing only, without adjustment for autocorrelation or multiple comparisons.

## Isolated decoding evidence

`decode_microbenchmark.json`: identical synthetic logits with nc80, reg_max16, N8400, K300, B1; cached anchors, 50 warmups, 200 randomized pairs, same CPU/thread/precision. Decode-all+top-k mean/median/P95 = 10.975/10.886/12.883 ms; select-first = 7.741/7.654/9.244 ms. Paired mean saving 3.234 ms, within-run bootstrap95% [3.066,3.408]. This confirms a local computation saving, **not whole-detector acceleration**; convolution, feature extraction and their cache interactions are excluded. The calculated FP32 DFL probability tensor is 2,150,400 versus 76,800 bytes. These are tensor-size calculations, not peak process/GPU memory measurements.

## Tests and export

Two suites run successfully: 18 methods total, 16 pass and 2 CUDA skips. Existing OHR/fusion/SimAM regressions remain covered. New tests check synthetic nonuniform and saturated/tied logits, classes1/4/80, reg_max1/16, FP32/FP64, batch2, class-aware/agnostic selection, K>N eager, duplicated positions, unchanged non-end2end inference, exact full-model training loss/gradients/BN state, fused/unfused model outputs, CPU BF16 autocast, n/s/m/l/x parsing for both YAMLs and actual YOLO save/load. Both pre-edit stock YOLO11 and Orthogonal reference snapshots still match exactly. GPU/FP16 deployment remains unverified.

Official YOLO dynamic ONNX export, opset17, ONNX1.14.0, ORT1.15.1 CPU passes. Dynamic B/H/W outputs are finite at [1,3,128,160] and [2,3,160,192]. Raw dense logits agree at 1e-4 tolerance. A separate exported selected decoder is compared against the **original decode-all numerical oracle**, with separated class scores to avoid cross-backend top-k ties: all four batch/shape/agnostic combinations pass (largest absolute error 0.00006103515625, identical class indices). Full random-model post-top-k outputs receive shape/finite checks only because tied scores can rank differently between backends. Export still requires N>=max_det. TensorRT/INT8/CUDA are not validated. Export warning about general advanced indexing does not imply a failing graph here; our selection indices are nonnegative, and runtime comparisons passed.

## Decision and next hypothesis

**REVISE:** keep selected decoding as an explicit experiment, exclude it from the currently preferred Efficient configuration. It passes correctness and reduces isolated decoding cost, but there is insufficient evidence for a practical whole-model gain on this CPU. No parameter reduction, measured peak-memory reduction, mAP gain or scientific originality is established. Accuracy and paper readiness remain **CHƯA ĐƯỢC XÁC MINH BẰNG THỰC NGHIỆM.**

Next hypothesis: convolution and feature movement dominate deploy performance, so spending architectural capacity differently in the regression tower may matter more than postprocessing alone. That changes learned representations and needs a reproduced trained baseline, profiling on the intended deployment device, and controlled accuracy/latency ablation before adoption. Do not reduce channels or add attention on the basis of this microbenchmark.

## Reproduce

```powershell
.venv/Scripts/python.exe -B research/test_orthogonal_efficient.py
.venv/Scripts/python.exe -B research/test_selected_decode.py
.venv/Scripts/python.exe -B research/benchmark_orthogonal.py --selected-ablation --output experiments/orthogonal_selected/new_benchmark.json
.venv/Scripts/python.exe -B research/profile_selected_decode.py
.venv/Scripts/python.exe -B research/export_orthogonal_probe.py --selected
```

Timing excludes preprocessing, data loading, IO and visualization. Evidence includes raw timing samples, source/config hashes, environment, commands and the immutable original random-weight reference. Exported `efficient_untrained.pt` is a test artifact, not trained detection weights. No training curves, PR curves or confusion matrices are generated because no training/evaluation occurred.
