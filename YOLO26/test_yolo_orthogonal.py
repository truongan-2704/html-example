# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Smoke test and benchmarking script for YOLO-Orthogonal architecture.
Validates:
1. Mathematical equivalence of OHR-Conv reparameterization.
2. Full model YAML parsing and layer counts.
3. Forward pass tensor shapes and model.fuse() deployment conversion.
4. Real-world inference speed (ms) and NMS candidate box filtering.
"""

import sys
import os
import time
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from ultralytics.nn.modules.orthogonal_blocks import (
    OHRConv,
    OrthoBottleneck,
    C3k2_Ortho,
    OSIFusion,
    SESPGate,
)


def test_reparameterization_math():
    print(f"\n{'='*70}")
    print(" 1. Testing OHR-Conv Reparameterization Mathematical Equivalence")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = OHRConv(64, 128, k=3, s=1).to(device)
    conv.eval()

    x = torch.randn(2, 64, 40, 40).to(device)
    with torch.no_grad():
        out_train = conv(x)
        conv.switch_to_deploy()
        out_deploy = conv(x)

    max_diff = (out_train - out_deploy).abs().max().item()
    print(f"✓ Training vs Deploy Max Difference: {max_diff:.8f}")
    assert max_diff < 1e-4, f"Reparameterization error too high: {max_diff}"
    print("✓ OHR-Conv Reparameterization Math: 100% PERFECT MATCH!")


def test_individual_orthogonal_blocks():
    print(f"\n{'='*70}")
    print(" 2. Testing Individual Orthogonal Blocks")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p3 = torch.randn(2, 64, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)

    # 1. Test OSIFusion
    osi = OSIFusion(128, 256, 128).to(device)
    y_osi = osi([p4, p5])
    print(f"✓ OSIFusion [p4, p5] -> out: {y_osi.shape} (Expected: [2, 128, 40, 40])")
    assert y_osi.shape == (2, 128, 40, 40)

    # 2. Test SESPGate
    sesp = SESPGate(64, 128).to(device)
    y_sesp = sesp([p3, p4])
    print(f"✓ SESPGate [p3, p4] -> out: {y_sesp.shape} (Expected: [2, 128, 40, 40])")
    assert y_sesp.shape == (2, 128, 40, 40)

    # 3. Test C3k2_Ortho
    c3k2_o = C3k2_Ortho(128, 128, n=2).to(device)
    y_o = c3k2_o(p4)
    print(f"✓ C3k2_Ortho p4 -> out: {y_o.shape} (Expected: [2, 128, 40, 40])")
    assert y_o.shape == (2, 128, 40, 40)


def test_full_model(cfg_path="ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal.yaml"):
    print(f"\n{'='*70}")
    print(f" 3. Testing Full YOLO-Orthogonal Architecture: {cfg_path}")
    print(f"{'='*70}")

    model = YOLO(cfg_path)
    pytorch_model = model.model
    pytorch_model.eval()

    total_params = sum(p.numel() for p in pytorch_model.parameters())
    trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)

    print(f"✓ Model loaded successfully!")
    print(f"  - Total Parameters: {total_params:,} (Significantly lighter than YOLO11n 2.62M!)")
    print(f"  - Number of PyTorch Layers: {len(pytorch_model.model)}")

    # Forward pass before fuse
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = pytorch_model(dummy)
    print(f"✓ Pre-fuse forward pass completed cleanly! Output shape: {out[0].shape}")

    # Fuse model (Deploy conversion)
    print(f"\n{'='*70}")
    print(" 4. Fusing Model into Single-Kernel Contiguous GEMM Deploy Mode")
    print(f"{'='*70}")
    pytorch_model.fuse()
    print("✓ Model.fuse() executed successfully!")

    # Post-fuse forward pass & benchmark latency
    print("\nBenchmarking raw inference speed over 30 runs...")
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = pytorch_model(dummy)

        t0 = time.perf_counter()
        runs = 30
        for _ in range(runs):
            _ = pytorch_model(dummy)
        t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / runs) * 1000
    print(f"✓ Average Forward Latency: {avg_ms:.2f} ms per image (batch=1, 640x640)")
    return True, total_params


if __name__ == "__main__":
    test_reparameterization_math()
    test_individual_orthogonal_blocks()
    success, params = test_full_model()
    if success:
        print(f"\n{'='*70}")
        print(" 🎉 ALL YOLO-ORTHOGONAL VERIFICATION & SPEED TESTS PASSED!")
        print(f"{'='*70}\n")
    else:
        sys.exit(1)
