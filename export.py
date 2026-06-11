from ultralytics import YOLO

model = YOLO("models/yolo-pose/yolo26m-pose.pt")

model.export(
    format="onnx",
    half=True,
    device=0
)