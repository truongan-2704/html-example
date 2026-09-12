# YOLO11 bottleneck analysis — local CPU evidence

Ngày 2026-09-11. Trạng thái **PARTIAL**: đã profile baseline local trên CPU; chưa có CUDA kernel time, GPU peak memory, DRAM bandwidth, peak CPU activation memory hoặc end-to-end benchmark. Không gọi các trường unavailable là zero. Không có proposed model hoặc accuracy comparison.

## Phương pháp

Evidence: `experiments/exp000_audit/profile.json`, `cpu_operators.json`, `baseline_probe.json`; script `research/profile_baseline.py`. Model YOLO11n local, nc=80, seed 0, weights ngẫu nhiên, input 1×3×640×640, FP32, CPU một thread, eager eval, không data loading/NMS/visualization/IO trong timing. Python 3.9.13, torch 2.2.2+cpu, Windows, CPU identifier trong environment.txt. GPU NVIDIA GeForce RTX 4060 Laptop có mặt nhưng torch không có CUDA.

Count **2 FLOPs/MAC** cho Conv2d, thêm hai matmul trong Attention. Coverage loại BN/SiLU/pool/softmax/resize/concat/decode/NMS arithmetic; vì vậy gọi **Conv+matmul GFLOPs**, không gọi đây là tổng mọi FLOPs. Formula Conv = 2·B·Ho·Wo·Co·(Ci/groups)·kh·kw; Attention matmuls = 2·B·heads·N²·(key_dim+head_dim). Hooks không đếm lại convolution bên trong Attention khi cộng matmul.

Stage latency: 20 forward với pre/post hooks ở 24 top-level YAML stages, wall-clock CPU; mỗi stage bao gồm toàn bộ children. Không cộng stage timings vào model-only benchmark hoặc gọi là GPU kernels. Stage hooks có overhead và chưa có warmup riêng đủ dài: dùng để xác định vị trí khảo sát, không làm số publication.

Model-only latency: hooks đã gỡ, 10 warmup và 50 samples riêng cho unfused/fused. FPS = batch×1000/mean_ms. Fused model là deepcopy; baseline nguyên không đổi. Không có CUDA synchronize vì không dùng CUDA; khi chạy GPU phải thêm synchronize trước/sau timing và ghi CUDA events/profiler riêng.

## Parameter và compute distribution

| Region | Parameters | Conv+matmul GFLOPs | Tỷ lệ phép tính |
|---|---|---|---|
| Backbone | 1,365,472 | 3.099955 | 47.38% |
| Neck | 793,696 | 1.572864 | 24.04% |
| Head | 464,912 | 1.869683 | 28.58% |

Total parameters 2,624,080, Conv+matmul 6.5425024 GFLOPs. 16 DFL weights fixed được tính vào parameters. Không đổi nc=4 âm thầm trong bảng này: head width/class output sẽ khác.

## Top 10 stage theo FLOPs

| Index | Module | Conv+matmul GFLOPs |
|---|---|---|
| 23 | Detect | 1.869683 |
| 3 | Conv | 0.471859 |
| 5 | Conv | 0.471859 |
| 16 | C3k2 | 0.406323 |
| 13 | C3k2 | 0.353894 |
| 2 | C3k2 | 0.327680 |
| 4 | C3k2 | 0.327680 |
| 22 | C3k2 | 0.301466 |
| 6 | C3k2 | 0.275251 |
| 8 | C3k2 | 0.275251 |

Primitive Conv/matmul details không trùng lặp nằm trong `primitive_flops`. Hai Conv box P3 `model.23.cv2.0.0.conv` và `model.23.cv2.0.1.conv` mỗi cái 0.4718592 GFLOPs; tổng 0.9437184, khoảng 14.42% toàn bộ phép tính được đếm. Đây là target khảo sát cụ thể, chưa phải bằng chứng có thể bỏ bớt mà giữ mAP.

## Top 10 stage theo CPU wall time

| Index | Module | Mean ms có hooks |
|---|---|---|
| 23 | Detect | 120.316 |
| 2 | C3k2 | 42.886 |
| 9 | SPPF | 32.727 |
| 4 | C3k2 | 25.360 |
| 16 | C3k2 | 24.091 |
| 3 | Conv | 19.649 |
| 0 | Conv | 18.817 |
| 13 | C3k2 | 18.688 |
| 6 | C3k2 | 18.476 |
| 1 | Conv | 18.096 |

## Top 10 output storage footprints — proxy, không phải peak activation memory

| Index | Module | Output storage MiB |
|---|---|---|
| 23 | Detect | 10.040 |
| 0 | Conv | 6.250 |
| 2 | C3k2 | 6.250 |
| 15 | Concat | 6.250 |
| 1 | Conv | 3.125 |
| 4 | C3k2 | 3.125 |
| 14 | Upsample | 3.125 |
| 12 | Concat | 2.344 |
| 3 | Conv | 1.563 |
| 11 | Upsample | 1.563 |

Đếm storage duy nhất trong output mỗi stage, có tính underlying storage của views. Detect output chứa decoded/raw/feats; feats tham chiếu input nên 10.040 MiB của Detect không phải toàn bộ allocations mới của head. Không cộng các hàng để suy peak. Stage 0, 2, 15 cùng output footprint 6.25 MiB, nhưng liveness/temporaries khác nhau.

CPU profiler có self allocation deltas theo operator; deltas có thể âm, không phải peak. `aten::cat` tổng allocations trong profiled forward là 33.801 MiB; đây là tổng allocation events, **không phải** DRAM bytes hoặc live peak. Top-10 memory thật ở module và memory bandwidth **chưa đo**, không suy từ kích thước tensor.

## Model-only CPU timing, exploratory

| Model mode | Mean ms | Median ms | P95 ms | FPS | Parameters |
|---|---|---|---|---|---|
| unfused | 395.822 | 389.322 | 465.199 | 2.526 | 2,624,080 |
| fused | 390.030 | 389.982 | 448.780 | 2.564 | 2,616,248 |

Lượt probe trước (5 warmup, 30 samples, cùng loại CPU scope) có mean 146.921 ms, khác đáng kể lượt profile này. Chưa kiểm soát CPU clocks, power/thermal và hệ thống nền; unfused/fused được đo tuần tự, không randomize thứ tự. **Không kết luận fuse nhanh hơn một cách đáng tin cậy từ chênh lệch nhỏ này.** Cần đo lại trong điều kiện kiểm soát sau khi xác định target hardware. Đây không phải số throughput tối đa CPU nhiều thread.

End-to-end latency (preprocess + model + decode/NMS + output handling trong RAM) chưa đo. Disk IO/visualization phải báo riêng nếu dùng. Số mAP50, mAP50-95, precision, recall, APsmall/medium/large đều NOT_EVALUATED vì chưa có trained baseline được xác minh.

## Phân tích riêng

**Backbone.** Hai dense stride-2 conv layers 3/5 tốn MAC; early C3k2 có tensors lớn. SPPF có CPU time đáng chú ý dù conv FLOPs nhỏ: maxpool/activation không nằm trong counter chính. Không thể gọi layer memory-bound khi chưa có hardware counters.

**Neck.** Upsample rồi concat làm channel tensor lớn trước pointwise projection, nổi bật ở layer 15/16. Fusion optimization phải kiểm tra cả projection cost, feature retention và launches; concat zero params không zero runtime. Có thể khảo sát project-before-resize hoặc fusion subspace, nhưng phải so với FPN/BiFPN/ASFF và low-rank prior art.

**Head.** Box branch P3 dense Conv là target có cơ sở đo. Classification đã dùng depthwise; shared weights chỉ giảm parameters, không tự giảm MAC nếu vẫn evaluate trên mọi scale. Giảm reg_max có thể thay calibration/localization và mất compatibility; không đổi cùng lúc với backbone để tránh confounder.

**Vật nhỏ.** Stride nhỏ nhất 8; không có P2 output. Cần xác minh object size distribution và FN theo size trước khi tăng high-resolution branch. Dataset apple/vehicles đều không mặc định là COCO standard; APsmall phải được tính với annotation/evaluator conventions rõ ràng.

Các hướng trên là candidate để lập hypothesis, không phải architecture đã chọn. **CHƯA ĐƯỢC XÁC MINH BẰNG THỰC NGHIỆM.**

## Regression probes thực tế

- Snapshot state_dict load strict và input/output roundtrip: exact match.
- CPU batch 2, rectangular 64×96, targets khác lớp ở hai ảnh: forward, real TAL/loss, backward finite; 255 trainable parameter tensors có gradient.
- Standard fuse trên copy: decoded outputs trong rtol=atol=1e-4. Đây không phải checkpoint upstream compatibility test.
- Isolated SPPF activation probe, batch 2 và 19×21 feature: cùng weights nhưng Identity so SiLU có max absolute difference 0.654512. Chứng minh constructor difference có ảnh hưởng trên feature ngẫu nhiên, không đo ảnh hưởng mAP.
- 527 file source/config Ultralytics khớp SHA256 trước audit; không sửa baseline.
- CUDA/AMP CUDA: SKIPPED, không ghi PASS. Full custom-family/ONNX/TensorRT regression chưa chạy.

## Gate và việc còn thiếu

Cần xác định baseline khoa học: local 8.4.103 có SPPF khác tag v8.3.0, hoặc reproduction release trong môi trường tách biệt. Không sửa default local để giả làm release. Hoàn thiện nearest literature và numerical extraction trước chọn architecture; chuẩn hóa data/split/pretrained/budget trước training. Chưa đạt baseline accuracy reproduction gate; **dừng so sánh accuracy và quyết định ACCEPT/REVISE/REJECT của kiến trúc** cho đến khi có evidence.

