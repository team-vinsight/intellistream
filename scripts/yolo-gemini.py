import cv2
import time
import psutil
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import SGDClassifier

# 1. Initialize YOLO Models
models = {
    "n": YOLO("models/yolo/yolo26n.pt"),
    "s": YOLO("models/yolo/yolo26s.pt"),
    "m": YOLO("models/yolo/yolo26m.pt"),
}
model_levels = ["n", "s", "m"]
current_level = 0
current_model = models[model_levels[current_level]]

# 2. Setup Supervised Online ML Model
# SGDClassifier allows us to update the model frame-by-frame using partial_fit
ml_router = SGDClassifier(loss="log_loss", random_state=42)
classes = np.array([0, 1, 2]) # Corresponds to model_levels indices

# Pre-train the ML router with a few dummy baseline rules so it doesn't guess randomly at start
X_init = np.array([
    [10.0, 80.0, 30.0],  # Low CPU use, High RAM available, High Target FPS -> Pick Medium (2)
    [50.0, 50.0, 20.0],  # Moderate resources -> Pick Small (1)
    [90.0, 10.0, 10.0]   # Heavy CPU use, Low RAM -> Pick Nano (0)
])
y_init = np.array([2, 1, 0])
ml_router.partial_fit(X_init, y_init, classes=classes)

# Performance targets
TARGET_FPS = 15

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def get_system_features():
    """Extracts live resource metrics as features for the ML model."""
    cpu_pct = psutil.cpu_percent()
    ram_pct = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    return np.array([[cpu_pct, ram_pct, TARGET_FPS]])

# Frame control and warm up
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Get current resource metrics
    features = get_system_features()

    # 3. ML Model Inference: Predict the optimal model level based on resources
    # We switch models every 5 frames to avoid jittering/overhead lags
    if frame_count % 5 == 0:
        predicted_level = int(ml_router.predict(features)[0])
        if predicted_level != current_level:
            current_level = predicted_level
            current_model = models[model_levels[current_level]]
            print(f"🧠 ML Router shifted system to YOLO-{model_levels[current_level]}")

    # Run object detection
    results = current_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Calculate actual FPS performance
    end_time = time.time()
    inference_time = end_time - start_time
    actual_fps = 1 / inference_time if inference_time > 0 else 0

    # 4. Feedback Loop (Incremental Training)
    # Define what a 'good decision' looks like to feed back into the model
    if actual_fps >= TARGET_FPS and current_level < 2:
        # If resource footprint allows for higher performance than target, teach ML to scale up
        correct_label = np.array([current_level + 1])
        ml_router.partial_fit(features, correct_label)
    elif actual_fps < (TARGET_FPS - 5) and current_level > 0:
        # If system is choking and falling below performance targets, teach ML to scale down
        correct_label = np.array([current_level - 1])
        ml_router.partial_fit(features, correct_label)
    else:
        # Current model is performing stably for these resources; reinforce this decision
        correct_label = np.array([current_level])
        ml_router.partial_fit(features, correct_label)

    # Overlay Telemetry
    cv2.putText(
        annotated_frame,
        f"Model: YOLO-{model_levels[current_level]} | FPS: {actual_fps:.1f} | CPU: {features[0][0]}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("ML-Driven Adaptive YOLO", annotated_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()