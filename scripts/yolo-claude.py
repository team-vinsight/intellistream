import cv2
import time
import numpy as np
import psutil
import collections
import pickle
import os
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Try importing GPU monitoring (optional)
# ─────────────────────────────────────────────
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# ─────────────────────────────────────────────
# Adaptive Model Selector (Online Neural Net)
# ─────────────────────────────────────────────
class AdaptiveModelSelector:
    """
    A lightweight 2-layer neural network trained online via SGD.

    Inputs  (8 features):
        [0] cpu_percent          – system CPU usage (0–100)
        [1] ram_percent          – system RAM usage (0–100)
        [2] gpu_percent          – GPU usage (0–100), 0 if unavailable
        [3] gpu_mem_percent      – GPU memory usage (0–100), 0 if unavailable
        [4] current_fps          – inference FPS over last N frames
        [5] inference_time_ms    – last inference duration in ms
        [6] frame_complexity     – normalised Laplacian variance (scene detail)
        [7] current_model_idx    – current model tier (0=n, 1=s, 2=m)

    Output (3 classes): model tier to use — 0=n, 1=s, 2=m
    """

    # --- architecture -------------------------------------------------
    INPUT_DIM  = 8
    HIDDEN_DIM = 16
    OUTPUT_DIM = 3          # three YOLO tiers

    # --- online-learning hyper-params ---------------------------------
    LEARNING_RATE   = 0.01
    MOMENTUM        = 0.9
    MIN_SAMPLES     = 20    # warm-up: rule-based labels until we have this many
    REPLAY_SIZE     = 200   # experience replay buffer length
    BATCH_SIZE      = 16    # mini-batch drawn from replay buffer
    UPDATE_INTERVAL = 5     # train every N new samples

    # --- rule-based thresholds (used for initial label generation) ----
    UPGRADE_FPS   = 22
    DOWNGRADE_FPS = 10
    CPU_HEAVY     = 75      # % above which we prefer lighter model
    RAM_HEAVY     = 85

    WEIGHTS_PATH = "adaptive_selector_weights.pkl"

    def __init__(self):
        self._init_weights()
        self._init_velocity()         # momentum buffers

        # experience replay buffer  →  (features, label)
        self.replay_buffer = collections.deque(maxlen=self.REPLAY_SIZE)
        self.sample_count  = 0
        self.update_ticker = 0

        # normalisation stats (running mean / std, Welford)
        self.feat_mean = np.zeros(self.INPUT_DIM)
        self.feat_var  = np.ones(self.INPUT_DIM)
        self.feat_n    = 0

        # load saved weights if they exist
        self._load_weights()

    # ── weight init (Xavier) ──────────────────────────────────────────
    def _init_weights(self):
        scale1 = np.sqrt(2.0 / self.INPUT_DIM)
        scale2 = np.sqrt(2.0 / self.HIDDEN_DIM)
        self.W1 = np.random.randn(self.INPUT_DIM,  self.HIDDEN_DIM) * scale1
        self.b1 = np.zeros(self.HIDDEN_DIM)
        self.W2 = np.random.randn(self.HIDDEN_DIM, self.OUTPUT_DIM) * scale2
        self.b2 = np.zeros(self.OUTPUT_DIM)

    def _init_velocity(self):
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    # ── activation helpers ────────────────────────────────────────────
    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(float)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    # ── forward pass ──────────────────────────────────────────────────
    def _forward(self, x):
        """x: (INPUT_DIM,) → returns (probs, cache)"""
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self._softmax(z2)
        return probs, (x, z1, a1)

    # ── backward pass (cross-entropy loss) ────────────────────────────
    def _backward(self, probs, label, cache):
        x, z1, a1 = cache
        one_hot = np.zeros(self.OUTPUT_DIM)
        one_hot[label] = 1.0
        dz2 = probs - one_hot                        # (OUTPUT_DIM,)
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_grad(z1)
        dW1 = np.outer(x, dz1)
        db1 = dz1
        return dW1, db1, dW2, db2

    # ── SGD + momentum update ─────────────────────────────────────────
    def _apply_gradients(self, dW1, db1, dW2, db2):
        self.vW1 = self.MOMENTUM * self.vW1 + self.LEARNING_RATE * dW1
        self.vb1 = self.MOMENTUM * self.vb1 + self.LEARNING_RATE * db1
        self.vW2 = self.MOMENTUM * self.vW2 + self.LEARNING_RATE * dW2
        self.vb2 = self.MOMENTUM * self.vb2 + self.LEARNING_RATE * db2
        self.W1 -= self.vW1;  self.b1 -= self.vb1
        self.W2 -= self.vW2;  self.b2 -= self.vb2

    # ── Welford running normalisation ─────────────────────────────────
    def _update_normalisation(self, raw_feat):
        self.feat_n += 1
        delta = raw_feat - self.feat_mean
        self.feat_mean += delta / self.feat_n
        delta2 = raw_feat - self.feat_mean
        self.feat_var += delta * delta2

    def _normalise(self, raw_feat):
        std = np.sqrt(self.feat_var / max(self.feat_n, 1)) + 1e-8
        return (raw_feat - self.feat_mean) / std

    # ── rule-based label (used during warm-up & as training signal) ───
    def _rule_label(self, raw_feat, current_level, num_levels):
        fps        = raw_feat[4]
        cpu        = raw_feat[0]
        ram        = raw_feat[1]
        resource_pressure = cpu > self.CPU_HEAVY or ram > self.RAM_HEAVY
        if resource_pressure or fps < self.DOWNGRADE_FPS:
            return max(0, current_level - 1)
        elif fps > self.UPGRADE_FPS and not resource_pressure:
            return min(num_levels - 1, current_level + 1)
        else:
            return current_level

    # ── public: observe outcome & train ───────────────────────────────
    def observe(self, raw_feat: np.ndarray, current_level: int, num_levels: int):
        """
        Called every frame with fresh telemetry.
        Updates running stats, stores experience, triggers training.
        """
        self._update_normalisation(raw_feat)
        label = self._rule_label(raw_feat, current_level, num_levels)
        norm_feat = self._normalise(raw_feat)
        self.replay_buffer.append((norm_feat.copy(), label))
        self.sample_count += 1
        self.update_ticker += 1

        if (self.sample_count >= self.MIN_SAMPLES and
                self.update_ticker >= self.UPDATE_INTERVAL and
                len(self.replay_buffer) >= self.BATCH_SIZE):
            self._train_batch()
            self.update_ticker = 0

    def _train_batch(self):
        """Sample a mini-batch from replay buffer and do one SGD step."""
        indices = np.random.choice(len(self.replay_buffer),
                                   self.BATCH_SIZE, replace=False)
        dW1_acc = np.zeros_like(self.W1)
        db1_acc = np.zeros_like(self.b1)
        dW2_acc = np.zeros_like(self.W2)
        db2_acc = np.zeros_like(self.b2)

        for i in indices:
            feat, label = self.replay_buffer[i]
            probs, cache = self._forward(feat)
            dW1, db1, dW2, db2 = self._backward(probs, label, cache)
            dW1_acc += dW1;  db1_acc += db1
            dW2_acc += dW2;  db2_acc += db2

        scale = 1.0 / self.BATCH_SIZE
        self._apply_gradients(dW1_acc * scale, db1_acc * scale,
                               dW2_acc * scale, db2_acc * scale)

    # ── public: predict best model tier ───────────────────────────────
    def predict(self, raw_feat: np.ndarray) -> int:
        """Returns the recommended model tier index (0/1/2)."""
        if self.sample_count < self.MIN_SAMPLES:
            # warm-up: fall back to current_level embedded in raw_feat[7]
            return int(round(raw_feat[7]))
        norm_feat = self._normalise(raw_feat)
        probs, _ = self._forward(norm_feat)
        return int(np.argmax(probs))

    def confidence(self, raw_feat: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities over tiers (for HUD display)."""
        if self.sample_count < self.MIN_SAMPLES:
            p = np.zeros(self.OUTPUT_DIM)
            p[int(round(raw_feat[7]))] = 1.0
            return p
        norm_feat = self._normalise(raw_feat)
        probs, _ = self._forward(norm_feat)
        return probs

    # ── persistence ───────────────────────────────────────────────────
    def save_weights(self):
        data = dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                    feat_mean=self.feat_mean, feat_var=self.feat_var,
                    feat_n=self.feat_n, sample_count=self.sample_count)
        with open(self.WEIGHTS_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"[Selector] Weights saved → {self.WEIGHTS_PATH}")

    def _load_weights(self):
        if not os.path.exists(self.WEIGHTS_PATH):
            return
        try:
            with open(self.WEIGHTS_PATH, "rb") as f:
                data = pickle.load(f)
            self.W1 = data["W1"];  self.b1 = data["b1"]
            self.W2 = data["W2"];  self.b2 = data["b2"]
            self.feat_mean    = data["feat_mean"]
            self.feat_var     = data["feat_var"]
            self.feat_n       = data["feat_n"]
            self.sample_count = data["sample_count"]
            self._init_velocity()   # reset momentum on load
            print(f"[Selector] Loaded weights ({self.sample_count} prior samples)")
        except Exception as e:
            print(f"[Selector] Could not load weights: {e}. Starting fresh.")


# ─────────────────────────────────────────────
# Telemetry helpers
# ─────────────────────────────────────────────
def get_gpu_stats():
    if not GPU_AVAILABLE:
        return 0.0, 0.0
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100, gpus[0].memoryUtil * 100
    except Exception:
        pass
    return 0.0, 0.0


def frame_complexity(frame, scale=0.25):
    """
    Normalised Laplacian variance — proxy for scene detail / edge density.
    Higher value → more complex scene → heavier model may be worthwhile.
    Returns a value roughly in [0, 1] after soft clipping.
    """
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap   = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.tanh(lap / 500.0))   # soft-clip to ~(0,1)


# ─────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────
def draw_hud(frame, level_names, current_level, fps, probs,
             sample_count, min_samples, cpu, ram, gpu):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # semi-transparent panel
    panel_w, panel_h = 300, 185
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def txt(text, y, color=(220, 220, 220), scale=0.55, thickness=1):
        cv2.putText(frame, text, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv2.LINE_AA)

    mode = "WARM-UP (rule)" if sample_count < min_samples else "ML active"
    txt(f"Model: YOLO-{level_names[current_level]}   FPS: {fps:5.1f}", 32,
        color=(0, 230, 100), scale=0.65, thickness=2)
    txt(f"Selector: {mode}  [{sample_count} samples]", 58)
    txt(f"CPU: {cpu:4.1f}%   RAM: {ram:4.1f}%   GPU: {gpu:4.1f}%", 80)

    # confidence bar for each tier
    bar_x, bar_y0, bar_max, bar_h = 16, 100, 180, 14
    for i, name in enumerate(level_names):
        p = probs[i]
        color = (0, 200, 80) if i == current_level else (120, 120, 120)
        cv2.rectangle(frame,
                      (bar_x, bar_y0 + i * 26),
                      (bar_x + int(bar_max * p), bar_y0 + i * 26 + bar_h),
                      color, -1)
        cv2.rectangle(frame,
                      (bar_x, bar_y0 + i * 26),
                      (bar_x + bar_max, bar_y0 + i * 26 + bar_h),
                      (160, 160, 160), 1)
        txt(f"{name}: {p*100:4.1f}%",
            bar_y0 + i * 26 + bar_h - 1,
            color=(255, 255, 255), scale=0.45)

    txt("Press 'q' quit  |  's' save weights", h - 12,
        color=(140, 140, 140), scale=0.42)
    return frame


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def open_camera():
    """Try camera indices 0-3; return the first one that delivers a real frame."""
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        # Drain a few frames — many cameras output black/green frames on startup
        ok = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and frame.max() > 10:
                ok = True
                break
        if ok:
            print(f"[Camera] Opened camera index {idx}  "
                  f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
            return cap
        cap.release()
    return None


def main():
    # ── load YOLO models ──────────────────────────────────────────────
    # models = {
    #     "n": YOLO("models/yolo/yolo26n.pt"),
    #     "s": YOLO("models/yolo/yolo26s.pt"),
    #     "m": YOLO("models/yolo/yolo26m.pt"),
    # }
    models = {
        "n": YOLO("models/yolo/yolo26n.engine"),
        "s": YOLO("models/yolo/yolo26s.engine"),
        "m": YOLO("models/yolo/yolo26m.engine"),
    }
    model_levels = ["n", "s", "m"]
    current_level = 0
    current_model = models[model_levels[current_level]]

    # ── ML selector ───────────────────────────────────────────────────
    selector = AdaptiveModelSelector()

    # ── FPS smoothing (rolling window) ────────────────────────────────
    fps_window = collections.deque(maxlen=10)
    fps_smooth  = 0.0

    # ── video ─────────────────────────────────────────────────────────
    cap = open_camera()
    if cap is None:
        print("Error: Could not open any webcam (tried indices 0-3).")
        return

    # Force a sensible capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WIN = "Adaptive YOLO — ML Selector"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    print("Adaptive YOLO with ML selector — press 'q' to quit, 's' to save.")
    consecutive_bad = 0

    while True:
        ret, frame = cap.read()

        # ── guard: skip genuinely blank / corrupt frames ───────────────
        if not ret or frame is None or frame.size == 0 or frame.max() <= 10:
            consecutive_bad += 1
            if consecutive_bad > 30:
                print("Error: Camera stopped delivering frames. Exiting.")
                break
            # show a placeholder so the window stays alive
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera...",
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 200, 200), 2, cv2.LINE_AA)
            cv2.imshow(WIN, placeholder)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue
        consecutive_bad = 0

        # ── inference ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = current_model(frame, verbose=False)
        t1 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        fps_window.append(1000.0 / inf_ms if inf_ms > 0 else 0)
        fps_smooth = float(np.mean(fps_window))

        # ── system telemetry ──────────────────────────────────────────
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent
        gpu_pct, gpu_mem_pct = get_gpu_stats()
        complexity = frame_complexity(frame)

        raw_feat = np.array([
            cpu_pct,
            ram_pct,
            gpu_pct,
            gpu_mem_pct,
            fps_smooth,
            inf_ms,
            complexity,
            float(current_level),
        ], dtype=np.float32)

        # ── ML predict & observe ──────────────────────────────────────
        recommended = selector.predict(raw_feat)
        probs        = selector.confidence(raw_feat)
        selector.observe(raw_feat, current_level, len(model_levels))

        # ── switch model if recommendation changed ────────────────────
        if recommended != current_level:
            direction = "⬆" if recommended > current_level else "⬇"
            current_level = recommended
            current_model = models[model_levels[current_level]]
            print(f"{direction} ML selector → YOLO-{model_levels[current_level]}"
                  f"  (FPS={fps_smooth:.1f}, CPU={cpu_pct:.0f}%, "
                  f"samples={selector.sample_count})")

        # ── draw & display ────────────────────────────────────────────
        annotated = results[0].plot()

        # Guard: results[0].plot() can return None on a bad inference frame
        if annotated is None or annotated.size == 0:
            annotated = frame.copy()

        # Resize to a consistent display size so the window never appears tiny
        display_h, display_w = annotated.shape[:2]
        target_w = 1280
        if display_w != target_w:
            scale = target_w / display_w
            annotated = cv2.resize(annotated, (target_w, int(display_h * scale)),
                                   interpolation=cv2.INTER_LINEAR)

        annotated = draw_hud(
            annotated, model_levels, current_level,
            fps_smooth, probs,
            selector.sample_count, selector.MIN_SAMPLES,
            cpu_pct, ram_pct, gpu_pct,
        )
        cv2.imshow(WIN, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            selector.save_weights()

    selector.save_weights()   # auto-save on clean exit
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()