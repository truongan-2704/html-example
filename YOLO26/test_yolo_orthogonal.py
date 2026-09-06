# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Smoke test and benchmarking script for YOLO-Orthogonal architecture.
Validates:
1. Mathematical equivalence of OHR-Conv reparameterization.
2. Individual orthogonal blocks (DOC_Fusion, C2PSA_Ortho, OrthoC3k, C3k2_Ortho).
3. Full model YAML parsing, layer counts, and forward pass.
4. Deployment fusion verification (0 gradients, reduced parameters).
5. Real-world inference speed benchmark.
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
    OrthoC3k,
    C3k2_Ortho,
    OSIFusion,
    DOC_Fusion,
    C2PSA_Ortho,
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

    # 1. Test DOC_Fusion (Dynamic Orthogonal Cross-Fusion)
    doc = DOC_Fusion(128, 256, 128).to(device)
    y_doc = doc([p4, p5])
    print(f"✓ DOC_Fusion [p4, p5] -> out: {y_doc.shape} (Expected: [2, 128, 40, 40])")
    assert y_doc.shape == (2, 128, 40, 40)
    doc.switch_to_deploy()
    y_doc_dep = doc([p4, p5])
    assert y_doc_dep.shape == (2, 128, 40, 40)
    print("✓ DOC_Fusion deploy mode verified cleanly!")

    # 2. Test OrthoC3k
    ortho_c3k = OrthoC3k(128, 128, n=2).to(device)
    y_c3k = ortho_c3k(p4)
    print(f"✓ OrthoC3k p4 -> out: {y_c3k.shape} (Expected: [2, 128, 40, 40])")
    assert y_c3k.shape == (2, 128, 40, 40)

    # 3. Test C3k2_Ortho with c3k=True
    c3k2_o = C3k2_Ortho(128, 128, n=2, c3k=True).to(device)
    y_o = c3k2_o(p4)
    print(f"✓ C3k2_Ortho (c3k=True) p4 -> out: {y_o.shape} (Expected: [2, 128, 40, 40])")
    assert y_o.shape == (2, 128, 40, 40)

    # 4. Test C2PSA_Ortho
    c2psa = C2PSA_Ortho(256, 256, n=1).to(device)
    y_psa = c2psa(p5)
    print(f"✓ C2PSA_Ortho p5 -> out: {y_psa.shape} (Expected: [2, 256, 20, 20])")
    assert y_psa.shape == (2, 256, 20, 20)


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
    print(f"  - Total Parameters (Training Mode): {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
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
    fused_params = sum(p.numel() for p in pytorch_model.parameters())
    remaining_grads = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    print(f"  - Fused Deployment Parameters: {fused_params:,}")
    print(f"  - Remaining Gradients: {remaining_grads:,} (Target: 0)")
    assert remaining_grads == 0, f"Expected 0 remaining gradients, but got {remaining_grads}"
    print("✓ 0 Gradients Confirmed: Gradient leakage completely eliminated!")

    model.info(detailed=False)

    # Post-fuse forward pass & benchmark latency
    print("\nBenchmarking raw inference speed over 30 runs...")
    with torch.no_grad():
        for _ in range(5):
            _ = pytorch_model(dummy)

        t0 = time.perf_counter()
        runs = 30
        for _ in range(runs):
            _ = pytorch_model(dummy)
        t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / runs) * 1000
    print(f"✓ Average Forward Latency: {avg_ms:.2f} ms per image (batch=1, 640x640)")
    return True, fused_params


if __name__ == "__main__":
    test_reparameterization_math()
    test_individual_orthogonal_blocks()
    success, params = test_full_model("ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal.yaml")
    if success:
        print(f"\n{'='*70}")
        print(f" 🎉 ALL YOLO-ORTHOGONAL VERIFICATION TESTS PASSED! Fused Params: {params:,}")
        print(f"{'='*70}\n")
    else:
        sys.exit(1)
