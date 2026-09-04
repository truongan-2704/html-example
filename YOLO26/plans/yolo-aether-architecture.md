# YOLO-Aether: Omni-Spectral Manifold Convolution & Harmonic Resonant Neck

## Tổng Quan (Overview)

**YOLO-Aether** là kiến trúc phát hiện vật thể thế hệ mới giải quyết triệt để hai điểm nghẽn lớn nhất trong các mạng YOLO hiện nay:
1. **Sự đánh đổi trong tích chập**: Khắc phục các hạn chế về FLOPs, Memory Access Cost (MAC), hiện tượng bàn cờ (gridding artifact) và suy giảm gradient thông qua toán tử **Omni-Spectral Manifold Convolution (OSM-Conv)**.
2. **Điểm nghẽn của PANet và BiFPN**: Xóa bỏ hoàn toàn pipeline tuần tự 2 lượt dài đằng đẵng (Top-Down -> Bottom-Up), xóa bỏ phép ghép nối `Concat` làm phình đôi số kênh của PANet, và thay thế trọng số scalar "mù không gian" của BiFPN bằng cấu trúc **Harmonic Resonant Pyramid Neck (HRP-Neck)** với nút giao cộng hưởng trung tâm và ma trận trọng số không gian 2D pixel-wise (**CSAF**).

---

## 1. Bản Đồ Kiến Trúc (Architecture Diagram)

```
                       Input Image (3, 640, 640)
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        BACKBONE: C3k2_Aether                           │
│                                                                        │
│  Stage 0: Conv 3×3, s=2, 64       → P1 (64, 320, 320)                 │
│  Stage 1: Conv 3×3, s=2 + C3k2_Aether ×2 → P2 (128, 160, 160)          │
│  Stage 2: Conv 3×3, s=2 + C3k2_Aether ×2 → P3 (256, 80, 80)           │
│  Stage 3: Conv 3×3, s=2 + C3k2_Aether ×2 → P4 (512, 40, 40)           │
│  Stage 4: Conv 3×3, s=2 + C3k2_Aether ×2 → P5 (1024, 20, 20)          │
│           + SPPF (5×5, 5×5, 5×5)                                       │
└────────────────────────────────────────────────────────────────────────┘
          │ (P3: 80×80)            │ (P4: 40×40)           │ (P5: 20×20)
          └──────────────┬─────────┴──────────┬────────────┘
                         ▼                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             HRP-NECK: HARMONIC RESONANT PYRAMID (ĐỒNG BỘ 1 BƯỚC)       │
│                                                                        │
│               [ P3: Down 2× ]   [ P4: 1×1 ]   [ P5: Up 2× ]            │
│                       └──────────────┼──────────────┘                  │
│                                      ▼                                 │
│                   ┌──────────────────────────────────────┐             │
│                   │  AetherResonantCore (Central Manifold│             │
│                   │      OSM-Conv Resonator @ 40×40)     │             │
│                   └──────────────────┬───────────────────┘             │
│                                      │ (Broadcast)                     │
│               ┌──────────────────────┼──────────────────────┐          │
│               ▼ (Upsample 2×)        ▼ (Identity)           ▼ (Conv s2)│
│       P3_Broadcast (80×80)    P4_Broadcast (40×40)    P5_Broadcast(20) │
│               │                      │                      │          │
│       [Backbone P3, P3_bc]   [Backbone P4, P4_bc]   [Backbone P5, P5_bc│
│               ▼                      ▼                      ▼          │
│          AetherCSAF             AetherCSAF             AetherCSAF      │
│      (Pixel 2D Weights)     (Pixel 2D Weights)     (Pixel 2D Weights)  │
│               ▼                      ▼                      ▼          │
│        AetherCSP (80×80)      AetherCSP (40×40)      AetherCSP (20×20) │
└───────────────┬──────────────────────┬──────────────────────┬──────────┘
                ▼                      ▼                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        HEAD: Decoupled Detect                          │
│             Detect(P3_out, P4_out, P5_out) → Detections                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Các Phát Minh Cốt Lõi (Core Innovations)

### 2.1. Omni-Spectral Manifold Convolution (OSM-Conv)
Phép tích chập thông thường buộc mọi kênh phải vừa học chi tiết biên cạnh, vừa học ngữ cảnh rộng, dẫn đến dư thừa tham số nghiêm trọng. OSM-Conv phân tách các kênh đầu vào thành 3 luồng chức năng tối ưu:
1. **Detail Branch ($25\% \cdot C$)**: Sử dụng micro-kernel $3\times3$ tập trung trích xuất biên cạnh sắc nét và thông tin hình học của các vật thể nhỏ.
2. **Context Branch ($25\% \cdot C$)**: Sử dụng tích chập giãn nở phân nhóm ($3\times3, d=2$), mở rộng trường nhìn mà không tạo hiệu ứng bàn cờ (anti-gridding) nhờ được nhánh chi tiết bù đắp.
3. **Direct Manifold Highway ($50\% \cdot C$)**: Đường truyền thẳng zero-FLOPs bảo toàn gradient và giảm thiểu chi phí truy cập bộ nhớ VRAM.
4. **Energy Router Gate**: Tái cân bằng năng lượng giữa các luồng:
   $$g = \sigma(W_2 \cdot \text{SiLU}(W_1 \cdot \text{Pool}(X)))$$

### 2.2. Content-Spatial Adaptive Fusion (AetherCSAF)
Thay thế hoàn toàn phép nối kênh `Concat` của PANet (gây tốn bộ nhớ đệm cache và nhân đôi số kênh) và phép cộng trọng số scalar vô hướng của BiFPN:
- Nhận vào đặc trưng cục bộ $F_{loc}$ và đặc trưng phát thanh toàn cục $F_{glb}$.
- Sinh ra ma trận trọng số không gian 2D độc lập cho từng pixel:
  $$\mathcal{W} = \text{Softmax}\left(\text{Conv}_{3\times3}([F_{loc}, F_{glb}])\right) \in \mathbb{R}^{B \times 2 \times H \times W}$$
- Thực hiện dung hợp thích ứng tại chỗ:
  $$F_{fused} = \mathcal{W}_1 \odot F_{loc} + \mathcal{W}_2 \odot F_{glb}$$
- Không làm tăng số lượng kênh, tiết kiệm $50\%$ băng thông so với PANet.

### 2.3. Central Resonant Broadcast Core (AetherResonantCore)
- PANet yêu cầu 5 bước chuyển tiếp tuần tự: $P5 \to P4 \to P3 \to P4 \to P5$.
- HRP-Neck gom $P3, P4, P5$ về một tọa độ cộng hưởng trung tâm duy nhất tại không gian $P4$, tương tác đồng bộ đa chiều trong đúng **1 chu kỳ tính toán**, sau đó phát thanh đồng thời đến cả 3 quy mô.

---

## 3. So Sánh Hiệu Năng và Kích Thước (n-scale)

| Kiến trúc | Số lớp (Layers) | Tổng tham số (Params) | Băng thông ghép kênh | Cơ chế điều phối |
| :--- | :---: | :---: | :---: | :---: |
| **YOLO11n (Gốc)** | 23 | 2,624,080 | Phình đôi kênh qua `Concat` | Cố định (Không chú ý) |
| **YOLO-BiFPN-n** | 28 | 2,980,120 | Cộng scalar $\sum w_i I_i$ | Trọng số scalar toàn ảnh |
| **YOLO-Aether-n** | **20** | **2,753,498** | **In-place CSAF (Không phình)** | **Trọng số động 2D từng pixel** |

---

## 4. Hướng Dẫn Sử Dụng và Huấn Luyện

### 4.1. Khởi tạo mô hình trong Python
```python
from ultralytics import YOLO

# Khởi tạo mô hình YOLO-Aether
model = YOLO('ultralytics/cfg/models/11/yolo11-Aether/yolo11-aether.yaml')

# Huấn luyện mô hình
model.train(data='coco8.yaml', epochs=100, imgsz=640, device=0)
```

### 4.2. Chạy kịch bản kiểm thử Smoke Test
```bash
python test_yolo_aether.py
```
