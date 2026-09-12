# Research status — 2026-09-12

## Current user priority: capacity before additional compression

The user now requires only a model lighter than **stock YOLO11**, while retaining learning capacity. `yolo11-orthogonal-capacity.yaml` supersedes B+C ratio0.25 as the recommended training candidate. It restores full spatial operators and increases hidden widths in P3/P4/P5 refinement. At nc80,n: training2,587,288 < stock2,624,080; deploy1,959,072 < stock2,616,248. The training runner defaults to Capacity; `--variant yolo11` selects stock baseline, while historical `--variant baseline` still selects original Orthogonal. See `ORTHOGONAL_CAPACITY.md`. The older rank results below remain historical experiments, not the current recommended compression level. Accuracy improvement remains unverified.

## Latest: actual Orthogonal-Rank architecture

Implemented CenterRankConv, C3k2_OrthoRank, OrthoRankDetect, all eight A/B/C YAMLs and rank sensitivity configurations. Selected B+C (ratio0.25) changes five downsamplers and the regression towers, preserving original CSP/neck fusion and classification. Deploy parameters1,813,504 ->1,141,984; Conv-only GFLOPs5.5642752 ->3.7808768; confirmation CPU latency133.911 ->122.442 ms at640, nc80. Separate nc4 benchmark also improves latency. Eight architecture tests pass, one CUDA skip; dynamic ONNX passes. A real-data one-epoch smoke run for baseline and B+C completes train/validation/checkpoint reload; subset mAP is0 for both, **not baseline reproduction or accuracy evidence**. Full-dataset multi-seed accuracy remains unverified. See `ORTHOGONAL_RANK_ARCHITECTURE.md` and `experiments/orthogonal_rank/`. Use `train_orthogonal_rank.py` for the new architecture; original `train.py` remains unchanged.

## Latest: selected decoding experiment

Added an opt-in `SelectDecodeDetect` subclass and two independent ablation YAMLs. Both original and Efficient configurations remain unchanged. New and existing suites: 16 passed, 2 CUDA skips. Dynamic ONNX selected-decoder output agrees with the original numerical oracle (maximum absolute coordinate error 0.0000611). Two randomized CPU model benchmarks did **not** establish an incremental speed benefit over Efficient, despite lower isolated DFL/decode cost. Decision: **REVISE**, retain as experimental ablation only. See `ORTHOGONAL_DECODE_STUDY.md`, `NOVELTY_MATRIX.md` and `experiments/orthogonal_selected/` for all results, limitations and source provenance. No trained mAP, peak GPU memory, cross-dataset superiority or paper-level contribution is established.

The sections below retain prior milestones. Earlier timing measurements were taken in another session and must not be treated as directly comparable with the latest measurements.

## Cập nhật sau yêu cầu cải tiến Orthogonal

Đã triển khai biến thể `yolo11-orthogonal-efficient.yaml` với eval projection trước upsample; training path và parameter keys được giữ tương thích. Sửa hai lỗi đã tái hiện: OHR fuse mất dtype/device và SimAM NaN ở feature 1×1. 10 tests pass, 1 CUDA skip; CPU paired benchmark và ONNX checks có artifacts. Stock YOLO11 và Orthogonal snapshot trước sửa vẫn khớp chính xác. Xem `ORTHOGONAL_IMPROVEMENT.md` cho patch, cơ sở toán học, prior art, số liệu và giới hạn. Chưa có training/mAP hoặc claim universal/paper-level superiority.

Phần dưới lưu trạng thái **audit ban đầu trước yêu cầu Orthogonal**, không phải trạng thái implementation mới nhất.

## Kết quả audit ban đầu

Đã hoàn thành mốc khởi đầu: xác định YOLO11 implementation local, phân tích detection execution path và tạo `CODEBASE_ANALYSIS.md` trước khi bắt đầu literature. Chưa hoàn thành dự án kiến trúc/thực nghiệm toàn bộ 19 phases. Không có module hay model YAML mới; không sửa source/config gốc.

| Phase | Trạng thái | Evidence / việc còn thiếu |
|---|---|---|
| 0 audit | COMPLETE cho detection path local; không chứng nhận mọi subsystem | CODEBASE_ANALYSIS.md, source inventory, stage shapes, forward snapshot |
| 1 literature | PARTIAL | 54 nguồn/model có link và cơ chế; nhiều metric NE, còn full-text extraction và rà 2025–2026/journals |
| 2 bottleneck | PARTIAL, exploratory CPU | YOLO11_BOTTLENECK_ANALYSIS.md, profile.json, cpu_operators.json; CUDA/kernel/bandwidth/peak activation chưa đo |
| 3–6 hypotheses/design/math/implementation | NOT STARTED | Chưa chốt architecture khi nearest work và baseline chưa đủ rõ |
| 7 proposed-module tests | NOT APPLICABLE YET | Baseline probes đã pass; không có proposed modules để test |
| 8 baseline reproduction | NOT REPRODUCED | Chỉ random-weight probes, chưa trained weights/provenance hoặc accuracy |
| 9–14 ablation/fair benchmark/complexity/Pareto/errors/seeds | NOT RUN | Không có bảng comparison được xác minh |
| 15 paper readiness | NOT READY | Chưa có contributions với evidence; không claim novelty/SOTA |
| 16 reproducibility | STARTED | Scripts, source hashes, environment, raw samples, snapshot và dataset split hashes |
| 17 quality | BASELINE PRESERVED | Scripts tách trong research/, source gốc unchanged |
| 18 regression | PARTIAL PASS | Exact local roundtrip, fuse tolerance, real loss/backward CPU; CUDA/export/model families chưa test |
| 19 final decision | NOT EVALUATED | Chưa có candidate/thí nghiệm; không gọi ACCEPT/REVISE/REJECT như đã đánh giá |

## Findings và decisions

1. `train.py` chọn Orthogonal, còn package local tự khai báo 8.4.103. Baseline phải ghi rõ YAML/scale/revision; không gọi script hiện tại là baseline YOLO11.
2. SPPF local tắt activation cv1 khác tag upstream v8.3.0. Probe cùng weights trên copy xác nhận feature outputs khác. Giữ nguyên local behavior. Muốn reproduce lịch sử phải dùng môi trường/source riêng và kiểm tra đầy đủ, không âm thầm sửa SPPF.
3. `.venv` dùng torch 2.2.2+cpu. NVIDIA RTX4060 Laptop có mặt nhưng CUDA/AMP không thể test trong môi trường này. Chưa thay package hay cài CUDA build.
4. Head box P3 và high-resolution neck là targets có evidence compute. Chưa có evidence có thể giảm budget mà giữ accuracy. CPU timing thay đổi lớn giữa lượt đo; không dùng chênh lệch nhỏ làm claim.
5. Literature có prior art rất gần cho DW/PConv, rep branches, weighted fusion, shared head, wavelets, frequency gates, training-only supervision. Chưa có novelty matrix vì chưa chọn computation pattern mới.
6. Người dùng đã được hỏi dataset chính và ngân sách training; chưa nhận câu trả lời tại thời điểm viết. Dataset apple được audit vì train.py đang dùng nó, không coi đây là protocol đã thống nhất.

## Dataset apple — kiểm tra thực tế

Nguồn evidence `experiments/exp000_audit/apple_dataset_audit.json`, chạy bằng `research/audit_dataset.py`. Đọc/hash tất cả ảnh và đọc tất cả label trong ba split; không thay dataset.

| Split | Images | Black Rot instances | Powdery_mildew | Rust | scab | Empty labels |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4,556 | 6,044 | 1,975 | 6,787 | 1,701 | 1 |
| Val | 570 | 695 | 272 | 722 | 251 | 0 |
| Test | 569 | 820 | 271 | 764 | 209 | 0 |

Không thấy missing labels, invalid five-field labels, exact byte duplicates cross-split hoặc filename-origin candidates cross-split. Không chứng minh annotation đúng, không perceptual dedup hoặc scene/subject leakage audit, chưa decode mọi ảnh để kiểm tra corruption. Không loại ảnh label rỗng vì có thể là negative hợp lệ. Không dùng test labels để tune kiến trúc; chỉ audit integrity.

## Reproduction commands

Chạy từ workspace bằng interpreter `.venv`; không cần pytest cho các executable assertions hiện có.

```powershell
.venv/Scripts/python.exe -B research/audit_baseline.py
.venv/Scripts/python.exe -B research/profile_baseline.py
.venv/Scripts/python.exe -B research/audit_dataset.py --data ultralytics/data/apple_leaft_detection/data.yaml --output experiments/exp000_audit/apple_dataset_audit.json
```

Hai script đầu ghi đè artifacts audit nếu chạy cùng output mặc định; để giữ reference ban đầu, dùng `--output` mới và `--reference` trỏ snapshot đúng. Đừng regenerate baseline reference sau khi sửa model rồi dùng nó làm mốc 'trước thay đổi'. Các file kiểm kê và environment ghi trạng thái tại lúc chạy, không tự cập nhật theo source tương lai.

## Next actions theo gate

Hoàn thiện review định lượng và nearest related works; chốt baseline local hay historical release, dataset/split, pretrained fairness và budget. Chuẩn bị môi trường CUDA tách biệt nếu GPU là target, xác nhận CPU/GPU/export compatibility và profile ổn định. Sau đó lập ít nhất 5 hypotheses, novelty matrix và mathematical proposal trước khi viết module. Khi có module phải qua unit/regression tests, rồi reproduce baseline accuracy trước mọi comparison/ablation. Không tạo PR/training curves/confusion matrices giả cho các experiment chưa chạy.

Tất cả lợi ích kiến trúc và khả năng paper-level contribution: **CHƯA ĐƯỢC XÁC MINH BẰNG THỰC NGHIỆM.**
