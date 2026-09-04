# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Smoke test script for YOLO-Aether model family.
Validates model instantiation from YAML config, exercises individual Aether blocks,
runs dummy forward passes, and verifies parameter counts and output shapes.
"""

import sys
import os
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from ultralytics.nn.modules.aether_blocks import (
    OSMConv,
    AetherBottleneck,
    C3k2_Aether,
    AetherCSP,
    AetherCSAF,
    AetherResonantCore,
)


def test_individual_blocks():
    print(f"\n{'='*70}")
    print(" Testing Individual YOLO-Aether Blocks")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Test OSMConv
    x = torch.randn(2, 64, 40, 40).to(device)
    osm = OSMConv(64, 128).to(device)
    y_osm = osm(x)
    print(f"✓ OSMConv: in {x.shape} -> out {y_osm.shape} (Expected: [2, 128, 40, 40])")
    assert y_osm.shape == (2, 128, 40, 40), f"Shape mismatch: {y_osm.shape}"

    # 2. Test AetherBottleneck
    btn = AetherBottleneck(128, 128).to(device)
    y_btn = btn(y_osm)
    print(f"✓ AetherBottleneck: in {y_osm.shape} -> out {y_btn.shape} (Expected: [2, 128, 40, 40])")
    assert y_btn.shape == (2, 128, 40, 40), f"Shape mismatch: {y_btn.shape}"

    # 3. Test C3k2_Aether
    c3k2_aether = C3k2_Aether(128, 128, n=2).to(device)
    y_c3k2 = c3k2_aether(y_btn)
    print(f"✓ C3k2_Aether: in {y_btn.shape} -> out {y_c3k2.shape} (Expected: [2, 128, 40, 40])")
    assert y_c3k2.shape == (2, 128, 40, 40), f"Shape mismatch: {y_c3k2.shape}"

    # 4. Test AetherResonantCore
    p3 = torch.randn(2, 64, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)
    res_core = AetherResonantCore([64, 128, 256], 128).to(device)
    y_core = res_core([p3, p4, p5])
    print(f"✓ AetherResonantCore: [p3, p4, p5] -> out {y_core.shape} (Expected: [2, 128, 40, 40])")
    assert y_core.shape == (2, 128, 40, 40), f"Shape mismatch: {y_core.shape}"

    # 5. Test AetherCSAF
    csaf = AetherCSAF([64, 128], 64).to(device)
    y_csaf = csaf([p3, y_core])
    print(f"✓ AetherCSAF: [p3(80x80), core(40x40)] -> out {y_csaf.shape} (Expected: [2, 64, 80, 80])")
    assert y_csaf.shape == (2, 64, 80, 80), f"Shape mismatch: {y_csaf.shape}"


def test_full_model(cfg_path="ultralytics/cfg/models/11/yolo11-Aether/yolo11-aether.yaml", img_size=(640, 640)):
    print(f"\n{'='*70}")
    print(f" Testing Full Architecture: {cfg_path}")
    print(f"{'='*70}")

    try:
        model = YOLO(cfg_path)
        pytorch_model = model.model
        pytorch_model.eval()

        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)

        print(f"✓ Model built successfully!")
        print(f"  - Total Parameters    : {total_params:,}")
        print(f"  - Trainable Parameters: {trainable_params:,}")
        print(f"  - Number of Layers    : {len(pytorch_model.model)}")

        dummy_input = torch.randn(1, 3, *img_size)
        with torch.no_grad():
            output = pytorch_model(dummy_input)

        if isinstance(output, (tuple, list)):
            print(f"  - Output elements: {len(output)}")
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"    - Output[{idx}] shape: {out.shape}")
                elif isinstance(out, (list, tuple)):
                    print(f"    - Output[{idx}] sub-elements: {len(out)}")
                    for sub_idx, sub_out in enumerate(out):
                        if isinstance(sub_out, torch.Tensor):
                            print(f"      - Output[{idx}][{sub_idx}] shape: {sub_out.shape}")
        elif isinstance(output, torch.Tensor):
            print(f"  - Output Tensor shape: {output.shape}")

        print(f"✓ Full forward pass completed cleanly!")
        return True, total_params
    except Exception as e:
        print(f"❌ FAILED to build/run {cfg_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


if __name__ == "__main__":
    test_individual_blocks()
    success, params = test_full_model()
    if success:
        print(f"\n{'='*70}")
        print(" 🎉 YOLO-AETHER ARCHITECTURE VERIFICATION PASSED!")
        print(f"{'='*70}\n")
    else:
        sys.exit(1)
