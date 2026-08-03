# YOLO26-Prism: End-to-End Frequency-Decomposed Detection Architecture

## Overview

**YOLO26-Prism** is the next-generation evolution of the **YOLO-Prism** architecture family, combining signal-processing frequency decomposition and moment-contrast attention with the structural advancements of **YOLO26** (End-to-End NMS-Free Training, Micro-bottleneck $e=0.25$ split ratios, and C2PSA global context aggregation).

---

## 1. Key Innovations

### 1.1 YOLO26 Structural Integration
- **End-to-End NMS-Free Prediction (`end2end: True`, `reg_max: 1`)**: Replaces standard NMS post-processing with end-to-end direct bounding box regression and one-to-one matching loss, reducing end-to-end inference latency.
- **Micro-Bottleneck Split Ratio ($e=0.25$)**: Applied in early backbone stages (P2, P3) to reduce parameter redundancy in high-resolution feature maps while retaining feature representation capacity.
- **C2PSA Global Context Tail**: Integrated at the end of P5 backbone (after `SPPF`) to capture long-range spatial dependencies using self-attention before passing features to the FPN top-down path.
- **Enhanced SPPF (`SPPF(1024, 5, 3, True)`)**: Uses residual shortcuts inside pooling stages for smoother multi-scale feature accumulation.

### 1.2 Prism Signal Processing Components
- **Dual-Frequency Decomposed Conv (DFDC / `DualFreqConv`)**: Splits channels into low-frequency ($K_{\text{lo}} \times K_{\text{lo}}$ DWConv) and high-frequency ($x - \text{AvgPool}(x) \to K_{\text{hi}} \times K_{\text{hi}}$ DWConv) subbands.
- **Tri-Band Frequency Decomposed Conv (TFDC / `TriFreqConv`)**: V2 extension introducing a mid-frequency band-pass filter ($\text{AvgPool}_{\text{fine}} - \text{AvgPool}_{\text{coarse}}$) for texture and pattern preservation.
- **Moment Contrast Gate (MCG / FCG)**: Channel attention mechanism based on Cauchy-Schwarz $L_2/L_1$ ratio ($\gamma$) and frequency variance ($\sigma^2$) to recalibrate activation concentration.
- **Frequency-Aware Spatial Refinement (FASR / AFR)**: Neck spatial attention utilizing low, mid, and high-frequency spatial energy maps with learnable per-level balance ($\alpha_{\text{lf}}, \alpha_{\text{mf}}, \alpha_{\text{hf}}$).

---

## 2. Model Family Summary

All models are located in `ultralytics/cfg/models/26/yolo26-prism/`:

| Model File | Description | Backbone | Neck | Output Heads | Parameters (n) |
|------------|-------------|----------|------|--------------|----------------|
| `yolo26-prism.yaml` | Core YOLO26-Prism V1 | `C3k2_Prism` ($e=0.25$) | `PrismCSP` (FASR) | P3, P4, P5 (300 det) | ~2.99M |
| `yolo26-prism-v2.yaml` | Tri-Band YOLO26-Prism V2 | `C3k2_PrismV2` ($e=0.25$) | `PrismV2CSP` (AFR) | P3, P4, P5 (300 det) | ~2.99M |
| `yolo26-prism-p2.yaml` | 4-Head Small Object Model | `C3k2_Prism` ($e=0.25$) | `PrismCSP` | P2, P3, P4, P5 | ~3.22M |
| `yolo26-prism-p6.yaml` | 4-Head High-Res Model | `C3k2_Prism` ($e=0.25$) | `PrismCSP` | P3, P4, P5, P6 | ~4.62M |

---

## 3. Usage & Training

### 3.1 Training in Python
```python
from ultralytics import YOLO

# Load YOLO26-Prism model
model = YOLO('ultralytics/cfg/models/26/yolo26-prism/yolo26-prism.yaml')

# Train on custom dataset
model.train(data='coco8.yaml', epochs=100, imgsz=640, device=0)
```

### 3.2 Verification
Run the verification harness:
```bash
python test_yolo26_prism.py
```
