import cv2
import time
from ultralytics import YOLO
from collections import deque

# -----------------------------
# Load models (PT or ENGINE)
# -----------------------------
models = {
    "n": YOLO("models/yolo-seg/yolo26n-seg.engine"),
    "s": YOLO("models/yolo-seg/yolo26s-seg.engine"),
    "m": YOLO("models/yolo-seg/yolo26m-seg.engine"),
}

model_levels = ["n", "s", "m"]

current_level = 0
current_model = models[model_levels[current_level]]

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 640
CONF = 0.5
IOU = 0.5

UPGRADE_FPS = 22
DOWNGRADE_FPS = 10

SWITCH_COOLDOWN = 2.0  # seconds

fps_history = deque(maxlen=10)
last_switch_time = 0

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# -----------------------------
# Warm-up (VERY IMPORTANT for .engine)
# -----------------------------
print("Warming up TensorRT engine...")
dummy = None
ret, dummy = cap.read()
if ret:
    for _ in range(30):
        _ = current_model.predict(
            dummy,
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            verbose=False
        )
print("Warm-up done.")

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()

    # -------------------------
    # Tracking instead of detection (stabilizes masks)
    # -------------------------
    results = current_model.track(
        source=frame,
        persist=True,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # -------------------------
    # FPS calculation (smoothed)
    # -------------------------
    inference_time = time.perf_counter() - start
    fps = 1.0 / inference_time if inference_time > 0 else 0

    fps_history.append(fps)
    smooth_fps = sum(fps_history) / len(fps_history)

    # -------------------------
    # Adaptive model switching (with cooldown)
    # -------------------------
    now = time.time()

    if now - last_switch_time > SWITCH_COOLDOWN:

        # upgrade
        if smooth_fps > UPGRADE_FPS and current_level < len(model_levels) - 1:
            current_level += 1
            current_model = models[model_levels[current_level]]
            last_switch_time = now
            print(f"⬆ Upgraded to {model_levels[current_level]}")

        # downgrade
        elif smooth_fps < DOWNGRADE_FPS and current_level > 0:
            current_level -= 1
            current_model = models[model_levels[current_level]]
            last_switch_time = now
            print(f"⬇ Downgraded to {model_levels[current_level]}")

    # -------------------------
    # UI overlay
    # -------------------------
    cv2.putText(
        annotated_frame,
        f"YOLO-SEG {model_levels[current_level]} | FPS: {smooth_fps:.1f}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Stable Adaptive YOLO Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()