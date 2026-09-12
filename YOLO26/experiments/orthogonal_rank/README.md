# Orthogonal-Rank architecture experiment

Current candidate: B+C, spatial rank ratio0.25. **REVISE pending full-dataset accuracy validation.** Unlike the earlier Efficient/selected-decode experiments, this changes trainable spatial kernels in downsamplers and regression towers.

Read `../../ORTHOGONAL_RANK_ARCHITECTURE.md` for equations, nearest work, actual tables and limitations. Use `../../train_orthogonal_rank.py` for baseline or candidate training; original `train.py` and original model YAML remain unchanged.

- `original_conv_profile.json`: measured original stage/convolution arithmetic distribution.
- `ablation_nc80_run1.json`: all eight A/B/C architectural combinations, raw CPU timings and initialization errors.
- `confirm_nc80_run2.json`, `confirm_nc4.json`: separate paired confirmation measurements at80/4 classes.
- `sensitivity_r0375.json`, `sensitivity_r05.json`: capacity/latency sensitivity, not trained accuracy tuning.
- `results.csv`: measured rows derived from the JSONs. Full-dataset mAP and GPU memory are unavailable in `metrics.json`.
- `tests_rank.txt`, `export_checks.json`, `export_log.txt`: architecture tests and dynamic ONNX validation.
- `rank_untrained.pt`, `rank_untrained.onnx`, `rank_raw_logits.onnx`, `weights/*random_fused_state.pt`: random-weight test/export artifacts, not useful trained detections.
- `serialized_sizes.json`: actual FP32 fused state_dict file sizes; not peak runtime memory or API full-checkpoint size.
- `initialization_nc80.json`: approximate SVD kernel-transfer errors, not task accuracy.
- `trainer_smoke/`: real-image one-epoch checks for baseline/B+C plus an SVD-initialized run. Tiny subset metrics do not constitute baseline reproduction. Source images/labels are copied, not changed.
- `training/prepared/`: exact full-dataset model/recipe configs. Preparing these files does not start full training.
- `source_snapshot.zip`, `source_hashes.json`, `tracked_changes.patch`, `git_commit.txt`, `environment.txt`, `packages.txt`: finalized code and provenance. Inference code is unchanged from measured runs; later changes concern validation and provenance. Per-run data/architecture/commands are also retained.

No meaningful multi-seed accuracy, APsmall, CUDA/TensorRT/INT8, peak GPU memory or end-to-end image-latency claim is made. CPU model-only timing excludes preprocessing and IO. Rank restriction can reduce representation capacity, so accuracy must be established before deployment selection or publication claims.
