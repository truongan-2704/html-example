# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Smoke test script for YOLO26-Prism model family.
Validates model instantiation from YAML configs, runs dummy forward passes,
and prints parameter counts & output tensor shapes.
"""

import sys
import os
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO


def test_model(cfg_path, img_size=(640, 640)):
    print(f"\n{'='*70}")
    print(f" Testing: {cfg_path}")
    print(f"{'='*70}")

    try:
        model = YOLO(cfg_path)
        pytorch_model = model.model
        pytorch_model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)

        print(f"✓ Model built successfully!")
        print(f"  - Total Parameters    : {total_params:,}")
        print(f"  - Trainable Parameters: {trainable_params:,}")
        print(f"  - Number of Layers    : {len(pytorch_model.model)}")

        # Run dummy forward pass
        dummy_input = torch.randn(1, 3, *img_size)
        with torch.no_grad():
            output = pytorch_model(dummy_input)

        if isinstance(output, (tuple, list)):
            print(f"  - Output tuple/list elements: {len(output)}")
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

        print(f"✓ Forward pass completed cleanly!")
        return True, total_params
    except Exception as e:
        print(f"❌ FAILED to build/run {cfg_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    configs = [
        "ultralytics/cfg/models/26/yolo26-prism/yolo26-prism.yaml",
        "ultralytics/cfg/models/26/yolo26-prism/yolo26-prism-v2.yaml",
        "ultralytics/cfg/models/26/yolo26-prism/yolo26-prism-p2.yaml",
        "ultralytics/cfg/models/26/yolo26-prism/yolo26-prism-p6.yaml",
    ]

    results = {}
    all_passed = True

    for cfg in configs:
        success, params = test_model(cfg)
        results[os.path.basename(cfg)] = (success, params)
        if not success:
            all_passed = False

    print(f"\n{'='*70}")
    print(" SUMMARY OF YOLO26-PRISM SMOKE TESTS")
    print(f"{'='*70}")
    for name, (success, params) in results.items():
        status = "PASSED ✓" if success else "FAILED ❌"
        print(f"  - {name:<30} : {status} ({params:,} params)")

    if all_passed:
        print("\n🎉 ALL YOLO26-PRISM MODELS PASSED TEST!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
