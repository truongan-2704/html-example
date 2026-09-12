# Audit evidence, not a training experiment

- `source_inventory.json`: all inventoried source/config/documentation hashes and Python AST symbols at initial audit time; original modules have not changed.
- `baseline_probe.json`: local YOLO11n config, 24 stage shapes, parameters, environment and raw CPU timings (untrained).
- `baseline_reference.pt`: tensor/config/state_dict snapshot for exact local regression. Random weights; not a pretrained or trained detector.
- `profile.json`: supported arithmetic counts, stage timings, output storage footprint proxies, uninstrumented CPU timings and executable assertion results.
- `cpu_operators.json`: one profiled CPU forward, self time/allocation deltas; not GPU counters, peak memory or bandwidth.
- `apple_dataset_audit.json`: resolved paths, class counts, exact image-hash overlap checks and split manifest hashes.
- `environment.txt`, `git_commit.txt`, `command.txt`: initial audit context. The Git root is the parent repository; hashes additionally identify local files.

No mAP, AP by size, precision/recall, training curves, PR curves, confusion matrices, ablations or multi-seed training results have been generated. Missing metrics are NOT_EVALUATED, not zero. See root reports for scope and caveats. Use the commands in RESEARCH_STATUS.md to reproduce, with a new output path to preserve these initial artifacts.
