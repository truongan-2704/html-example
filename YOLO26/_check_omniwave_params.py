import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO

print("="*60)
print("PARAMETER COMPARISON: YOLO11 vs OMNIWAVE-YOLO")
print("="*60)

for scale in ['n', 's']:
    print(f"\n--- Scale '{scale}' ---")
    # Baseline YOLO11
    base_model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
    base_model.model.yaml['scale'] = scale
    base_params = sum(p.numel() for p in base_model.model.parameters())

    # OmniWave
    ow_model = YOLO("ultralytics/cfg/models/11/yolo11-OmniWave/yolo11-omniwave.yaml")
    ow_model.model.yaml['scale'] = scale
    ow_params_pre = sum(p.numel() for p in ow_model.model.parameters())
    ow_model.fuse()
    ow_params_post = sum(p.numel() for p in ow_model.model.parameters())

    print(f"YOLO11-{scale} (Baseline):        {base_params:,} params ({base_params/1e6:.3f}M)")
    print(f"OmniWave-{scale} (Pre-fuse):      {ow_params_pre:,} params ({ow_params_pre/1e6:.3f}M)")
    print(f"OmniWave-{scale} (Post-fuse):     {ow_params_post:,} params ({ow_params_post/1e6:.3f}M)")
    diff = base_params - ow_params_post
    pct = (diff / base_params) * 100
    print(f"Reduction: -{diff:,} params (-{pct:.1f}% lighter than YOLO11!)")
