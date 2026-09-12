# Selected decoding execution ablation

See `../../ORTHOGONAL_DECODE_STUDY.md` for methods, measured tables, prior implementations and decision. **REVISE: experimental only.** Isolated decoding is faster, but two whole-model runs do not establish an incremental benefit over Efficient. There was no training or accuracy evaluation.

- `config.yaml`, `command.txt`, `environment.txt`, `git_commit.txt`, `source_hashes.json`, `tracked_changes.patch`: reproduction provenance. The Git commit alone does not include uncommitted research files; preserve the working tree matching the manifest. The patch records tracked-file changes only.
- `benchmark_640_run1.json`, `benchmark_640_run2.json`: complete model-only CPU timings and paired comparisons; `results.csv` is derived from these measured values.
- `decode_microbenchmark.json`: separate synthetic-logit decoding measurement; `decode_profile_log.txt` is its console output.
- `tests_fusion.txt`, `tests_selected.txt`: 16 passing methods, 2 unavailable-CUDA skips.
- `export_checks.json`, `export_log.txt`: official dynamic ONNX export, raw-logit comparisons and direct selected-decoder checks against original decoding.
- `metrics.json`: decision and explicit unavailable accuracy/memory metrics.
- `efficient_untrained.pt`, `efficient_untrained.onnx`, `efficient_raw_logits.onnx`, `selected_decoder_*.onnx`: random-weight/export test artifacts, **not trained detection models**.

Reference config/state/input remain in `../orthogonal_audit/before_changes.pt`; stock YOLO11 reference remains in `../exp000_audit/baseline_reference.pt`. They are not overwritten. No PR curve, training curve or confusion matrix is included because there was no corresponding experiment. Measured latency excludes preprocessing, data loading, IO and visualization. CPU runs do not establish GPU, edge-device or deployment-backend performance.
