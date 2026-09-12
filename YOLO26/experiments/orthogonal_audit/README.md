# Orthogonal engineering evaluation

All model checkpoints and exports here use RANDOM, UNTRAINED weights. They are not suitable for actual object detection or accuracy comparisons.

- `before_changes.pt`: preserved original Orthogonal config/state/input/output before patches.
- `benchmark.json`: raw paired model-only CPU timings and Conv2d arithmetic counts for fused original/efficient models.
- `efficient_untrained.pt`, `efficient_untrained.onnx`: checkpoint/official export smoke-test artifacts.
- `efficient_raw_logits.onnx`: raw logits numerical comparison graph, not a detection pipeline.
- `export_checks.json`: ONNX graph/runtime/numerical checks, including explicit top-k comparison limits.

Commands and all findings are documented in root `ORTHOGONAL_IMPROVEMENT.md`. Accuracy, real GPU performance and universal generalization are NOT_EVALUATED. Do not replace the before-change snapshot when checking regressions.
