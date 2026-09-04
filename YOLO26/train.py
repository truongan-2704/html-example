
from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO(r'ultralytics/cfg/models/26/yolo26.yaml')
    # model = YOLO(r'ultralytics/cfg/models/26/yolo26-prism/yolo26-prism.yaml')
    model = YOLO(r'ultralytics/cfg/models/11/yolo11-Aether/yolo11-aether.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal.yaml')
    # model = YOLO(r'ultralytics/cfg/models/26/yolo26-prism/yolo26-prism-v2.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-DCNF-V1Plus.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-IDC.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EfficientNetV2-CA.yaml')

    model.train(
        data=r'ultralytics/data/apple_leaft_detection/data.yaml',
        imgsz=640,
        batch=32,                         # Ổn định hơn, ít dao động mAP
        epochs=150,
        cache=False,
        amp=False,                        # FP32 cho độ chính xác cao nhất
        optimizer='SGD',
        patience=30,
        save_period=10,
        seed=42,
        project='runs/train',
        name='exp',
        workers=0,
        device='cpu',
        val=True
        # loss_type="aghiou",
    )