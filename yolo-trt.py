import cv2
import time
from ultralytics import YOLO

# Load multiple models
models = {
    "n": YOLO("models/yolo/yolo26n.engine"),
    "s": YOLO("models/yolo/yolo26s.engine"),
    "m": YOLO("models/yolo/yolo26m.engine"),
}

# Model order (from light → heavy)
model_levels = ["n", "s", "m"]
current_level = 0
current_model = models[model_levels[current_level]]

# Target performance thresholds (adjust as needed)
TARGET_FPS = 15
UPGRADE_FPS = 22   # if faster than this → use heavier model
DOWNGRADE_FPS = 10 # if slower than this → use lighter model

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Run inference
    results = current_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # FPS calculation
    end_time = time.time()
    inference_time = end_time - start_time
    fps = 1 / inference_time if inference_time > 0 else 0

    # Adaptive model switching
    if fps > UPGRADE_FPS and current_level < len(model_levels) - 1:
        current_level += 1
        current_model = models[model_levels[current_level]]
        print(f"⬆ Upgrading model to YOLO-{model_levels[current_level]}")

    elif fps < DOWNGRADE_FPS and current_level > 0:
        current_level -= 1
        current_model = models[model_levels[current_level]]
        print(f"⬇ Downgrading model to YOLO-{model_levels[current_level]}")

    # Overlay FPS + model info
    cv2.putText(
        annotated_frame,
        f"Model: {model_levels[current_level]} | FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Adaptive YOLO Webcam", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()