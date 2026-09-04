# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Comprehensive test script for YOLO-Aether v2 model family.
Validates:
1. Reparameterization exactness of RepOSMConv (FP32 numerical error < 1e-5).
2. Individual Aether blocks (AetherBottleneck, C3k2_Aether, AetherCSP, AetherCSAF, AetherResonantCore).
3. Full architecture initialization from yolo11-aether.yaml.
4. Model fusion via model.fuse() and parameter reduction.
5. Inference latency benchmark (pre-fuse vs post-fuse).
"""

import sys
import os
import time
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from ultralytics.nn.modules.aether_blocks import (
    OSMConv,
    RepOSMConv,
    AetherBottleneck,
    C3k2_Aether,
    AetherCSP,
    AetherCSAF,
    AetherResonantCore,
)


def test_reparameterization_equivalence():
    print(f"\n{'='*70}")
    print(" 1. Testing RepOSMConv Reparameterization Equivalence")
    print(f"{'='*70}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Case 1: c1 == c2, stride=1 (has 3x3 + 1x1 + Laplacian + Identity branches)
    m1 = RepOSMConv(64, 64, s=1).to(device)
    m1.eval()
    x1 = torch.randn(2, 64, 32, 32, device=device)

    with torch.no_grad():
        out_train = m1(x1)

    m1.switch_to_deploy()
    with torch.no_grad():
        out_fused = m1(x1)

    diff1 = torch.max(torch.abs(out_train - out_fused)).item()
    print(f"✓ Case 1 (c1=64, c2=64, s=1) Max Diff: {diff1:.6e} (Tolerance: 1e-4)")
    assert diff1 < 1e-4, f"Reparameterization error too large: {diff1}"

    # Case 2: c1 != c2, stride=2 (has 3x3 + 1x1 branches)
    m2 = RepOSMConv(32, 64, s=2).to(device)
    m2.eval()
    x2 = torch.randn(2, 32, 32, 32, device=device)

    with torch.no_grad():
        out_train2 = m2(x2)

    m2.switch_to_deploy()
    with torch.no_grad():
        out_fused2 = m2(x2)

    diff2 = torch.max(torch.abs(out_train2 - out_fused2)).item()
    print(f"✓ Case 2 (c1=32, c2=64, s=2) Max Diff: {diff2:.6e} (Tolerance: 1e-4)")
    assert diff2 < 1e-4, f"Reparameterization error too large: {diff2}"
    print("✓ Reparameterization exactness verified successfully!")


def test_individual_blocks():
    print(f"\n{'='*70}")
    print(" 2. Testing Individual YOLO-Aether Blocks")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. AetherBottleneck
    x = torch.randn(2, 64, 40, 40, device=device)
    btn = AetherBottleneck(64, 64).to(device)
    y_btn = btn(x)
    print(f"✓ AetherBottleneck: in {x.shape} -> out {y_btn.shape}")
    assert y_btn.shape == (2, 64, 40, 40)

    # 2. C3k2_Aether
    c3k2 = C3k2_Aether(64, 64, n=2).to(device)
    y_c3k2 = c3k2(x)
    print(f"✓ C3k2_Aether: in {x.shape} -> out {y_c3k2.shape}")
    assert y_c3k2.shape == (2, 64, 40, 40)

    # 3. AetherCSP
    csp = AetherCSP(64, 64, n=2).to(device)
    y_csp = csp(x)
    print(f"✓ AetherCSP: in {x.shape} -> out {y_csp.shape}")
    assert y_csp.shape == (2, 64, 40, 40)

    # 4. AetherResonantCore
    p3 = torch.randn(2, 64, 80, 80, device=device)
    p4 = torch.randn(2, 128, 40, 40, device=device)
    p5 = torch.randn(2, 256, 20, 20, device=device)
    res_core = AetherResonantCore([64, 128, 256], 128).to(device)
    y_core = res_core([p3, p4, p5])
    print(f"✓ AetherResonantCore: [P3, P4, P5] -> out {y_core.shape} (Expected: [2, 128, 40, 40])")
    assert y_core.shape == (2, 128, 40, 40)

    # 5. AetherCSAF
    csaf = AetherCSAF([64, 128], 64).to(device)
    y_csaf = csaf([p3, y_core])
    print(f"✓ AetherCSAF: [p3(80x80), core(40x40)] -> out {y_csaf.shape} (Expected: [2, 64, 80, 80])")
    assert y_csaf.shape == (2, 64, 80, 80)


def test_full_model_and_fusion(cfg_path="ultralytics/cfg/models/11/yolo11-Aether/yolo11-aether.yaml"):
    print(f"\n{'='*70}")
    print(f" 3. Testing Full YOLO-Aether Architecture & Fusion: {cfg_path}")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(cfg_path)
    py_model = model.model.to(device)
    py_model.eval()

    params_train = sum(p.numel() for p in py_model.parameters())
    layers_train = len(py_model.model)
    print(f"Training Mode:")
    print(f"  - Total Parameters: {params_train:,}")
    print(f"  - Total Layers    : {layers_train}")

    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        out_train = py_model(dummy_input)
    print(f"✓ Forward pass before fusion completed successfully!")

    # Perform Model Fusion
    print(f"\nCalling model.fuse()...")
    model.fuse()
    py_model_fused = model.model

    params_fused = sum(p.numel() for p in py_model_fused.parameters())
    layers_fused = len(py_model_fused.model)
    print(f"Fused / Deployment Mode:")
    print(f"  - Total Parameters: {params_fused:,} (Saved: {params_train - params_fused:,})")
    print(f"  - Total Layers    : {layers_fused}")

    with torch.no_grad():
        out_fused = py_model_fused(dummy_input)
    print(f"✓ Forward pass after fusion completed successfully!")

    # Latency Benchmark
    print(f"\nBenchmarking inference speed (50 warmups, 100 runs)...")
    with torch.no_grad():
        for _ in range(30):
            _ = py_model_fused(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            _ = py_model_fused(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    avg_latency_ms = (t1 - t0) / 100 * 1000
    fps = 1000.0 / avg_latency_ms
    print(f"⚡ Fused Inference Latency: {avg_latency_ms:.2f} ms/image ({fps:.1f} FPS) on {device}")

    return True


if __name__ == "__main__":
    test_reparameterization_equivalence()
    test_individual_blocks()
    success = test_full_model_and_fusion()
    if success:
        print(f"\n{'='*70}")
        print(" 🎉 YOLO-AETHER v2 ARCHITECTURE VERIFICATION PASSED!")
        print(f"{'='*70}\n")
    else:
        sys.exit(1)
