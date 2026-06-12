import cv2
import time
from ultralytics import YOLO

# Load pose models
models = {
    "n": YOLO("models/yolo-pose/yolo26n-pose.engine"),
    "s": YOLO("models/yolo-pose/yolo26s-pose.engine"),
    "m": YOLO("models/yolo-pose/yolo26m-pose.engine"),
}

model_levels = ["n", "s", "m"]

current_level = 0
current_model = models[model_levels[current_level]]

UPGRADE_FPS = 22
DOWNGRADE_FPS = 10

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()

    # Pose inference
    results = current_model.predict(
        source=frame,
        conf=0.25,
        verbose=False,
    )

    # Draw keypoints + skeletons
    annotated_frame = results[0].plot()

    inference_time = time.perf_counter() - start
    fps = 1.0 / inference_time if inference_time > 0 else 0

    # Adaptive model switching
    if fps > UPGRADE_FPS and current_level < len(model_levels) - 1:
        current_level += 1
        current_model = models[model_levels[current_level]]
        print(f"Upgraded to YOLO-{model_levels[current_level]}-pose")

    elif fps < DOWNGRADE_FPS and current_level > 0:
        current_level -= 1
        current_model = models[model_levels[current_level]]
        print(f"Downgraded to YOLO-{model_levels[current_level]}-pose")

    cv2.putText(
        annotated_frame,
        f"YOLO-Pose {model_levels[current_level]} | FPS: {fps:.1f}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Adaptive YOLO Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()