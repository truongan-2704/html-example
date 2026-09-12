# Current capacity-first candidate

Read `../../ORTHOGONAL_CAPACITY.md`. This supersedes the compact B+C recommendation following the user's change in priorities. The architecture restores full spatial operators and widens existing CSP refinement while remaining below measured stock YOLO11 parameter budgets.

- `validation_nc80.json`, `validation_nc4.json`: actual parameter/Conv counts, raw CPU timing samples and loss/backward/fuse checks. nc4 latency is noisy and does not show a mean-latency win; do not cherry-pick nc80 to claim universal speed.
- `export_checks.json`, `export_log.txt`: dynamic ONNX checks. `capacity_untrained*` and raw-logit ONNX files contain random weights, not trained detections.
- `trainer_smoke/`: real-data one-epoch integration test using the prior15/7-image subset, matching-class partial initialization, checkpoint reload and actual metrics (mAP0, not accuracy evidence).
- `training/prepared/`: full-data stock YOLO11 and Capacity recipes. Preparation does not execute full training.
- `source_snapshot.zip`, `source_hashes.json`: archived implementation at this milestone. Earlier experiments retain their own historical source snapshots.

The current entry point is `train_orthogonal_capacity.py`. `--variant yolo11` is stock YOLO11, `--variant baseline` retains the historical meaning of original Orthogonal, and default `--variant capacity` is the current candidate. Rank ablations remain available explicitly. Full training/accuracy and GPU performance remain unverified.
