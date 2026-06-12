import cv2
import time
import numpy as np
import psutil
import collections
import pickle
import os
from typing import Optional
from ultralytics import YOLO

# ─────────────────────────────────────────────
# GPU monitoring (optional)
# ─────────────────────────────────────────────
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Per-tier performance tracker  ←  the reward signal
# ─────────────────────────────────────────────────────────────────────────────
class TierPerformanceTracker:
    """
    Maintains an exponential moving average composite score for every model
    tier so the selector can learn *empirically* when a heavier model is
    worth the FPS cost.

    composite(tier) =
        FPS_W  × clip(fps / TARGET_FPS,  0, 1)
      + DET_W  × clip(avg_conf × log1p(n_det) / log1p(10),  0, 1)

    This encodes the trade-off the rule-based system cannot:
        "YOLO-M is worth it only when the scene has many confident detections."
    """

    EMA_ALPHA          = 0.08   # slow EMA — brief spikes should not dominate
    FPS_WEIGHT         = 0.55
    DET_WEIGHT         = 0.45
    TARGET_FPS         = 25.0
    MIN_OBS_FOR_LABEL  = 15     # need this many obs before trusting a tier's EMA
    SIGNIFICANT_DELTA  = 0.04   # must beat current tier by this to prefer it

    def __init__(self, n_tiers: int):
        self.n     = n_tiers
        self.ema   = [0.5] * n_tiers   # initialise to neutral 0.5
        self.count = [0]   * n_tiers

    def composite(self, fps: float, avg_conf: float, num_det: int) -> float:
        fps_score = float(np.clip(fps / self.TARGET_FPS, 0.0, 1.0))
        det_score = float(np.clip(
            avg_conf * np.log1p(num_det) / np.log1p(10), 0.0, 1.0))
        return self.FPS_WEIGHT * fps_score + self.DET_WEIGHT * det_score

    def update(self, tier: int, fps: float, avg_conf: float, num_det: int):
        s = self.composite(fps, avg_conf, num_det)
        if self.count[tier] == 0:
            self.ema[tier] = s
        else:
            self.ema[tier] = (1 - self.EMA_ALPHA) * self.ema[tier] + self.EMA_ALPHA * s
        self.count[tier] += 1

    def best_adjacent_tier(self, current: int) -> Optional[int]:
        """
        Return a neighbour tier (current ± 1) if it has enough observations
        AND significantly outscores the current tier.
        Returns None → stay at current tier.
        Only considers adjacent tiers to prevent large jumps (n → m directly).
        """
        best, best_score = None, self.ema[current]
        for candidate in (current - 1, current + 1):
            if not (0 <= candidate < self.n):
                continue
            if self.count[candidate] < self.MIN_OBS_FOR_LABEL:
                continue
            if self.ema[candidate] > best_score + self.SIGNIFICANT_DELTA:
                best, best_score = candidate, self.ema[candidate]
        return best


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive model selector — online MLP
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveModelSelector:
    """
    2-layer MLP (14 → 32 → 3) trained online via mini-batch SGD + momentum.

    ── Label generation (three phases) ───────────────────────────────────────
    1. Warm-up  (<MIN_SAMPLES):
         Pure rule-based labels only.
    2. Transition (MIN_SAMPLES → PERF_RAMP_SAMPLES):
         Linearly growing mix: rule labels + TierPerformanceTracker labels.
    3. Mature   (>PERF_RAMP_SAMPLES):
         Up to MAX_PERF_WEIGHT of labels come from observed performance,
         so the network learns *beyond* the hand-crafted rules.

    ── Input features (14) ────────────────────────────────────────────────────
      [0]  cpu_percent
      [1]  ram_percent
      [2]  gpu_percent
      [3]  gpu_mem_percent
      [4]  current_fps          10-frame rolling mean
      [5]  inference_time_ms
      [6]  frame_complexity     tanh-normalised Laplacian variance
      [7]  current_model_idx    0 / 1 / 2
      [8]  num_detections
      [9]  avg_confidence
      [10] max_confidence
      [11] track_count          detections with conf > 0.5  (reliable objects)
      [12] track_age            EMA of num_detections        (temporal stability)
      [13] motion_score         normalised frame-diff mean
    """

    INPUT_DIM  = 14
    HIDDEN_DIM = 32
    OUTPUT_DIM = 3

    LEARNING_RATE   = 0.01
    MOMENTUM        = 0.9
    MIN_SAMPLES     = 30
    REPLAY_SIZE     = 400
    BATCH_SIZE      = 24
    UPDATE_INTERVAL = 5

    MAX_PERF_WEIGHT   = 0.70    # ceiling on performance-based label fraction
    PERF_RAMP_SAMPLES = 300     # samples to ramp weight from 0 → MAX

    # rule-based thresholds (warm-up / fallback)
    UPGRADE_FPS   = 22
    DOWNGRADE_FPS = 10
    CPU_HEAVY     = 75
    RAM_HEAVY     = 85

    WEIGHTS_PATH = "adaptive_selector_weights-2.pkl"

    def __init__(self, n_tiers: int = 3):
        self.n_tiers = n_tiers
        self._init_weights()
        self._init_velocity()
        self.replay_buffer = collections.deque(maxlen=self.REPLAY_SIZE)
        self.sample_count  = 0
        self.update_ticker = 0
        # Welford online normalisation
        self.feat_mean = np.zeros(self.INPUT_DIM)
        self.feat_var  = np.ones(self.INPUT_DIM)
        self.feat_n    = 0
        # Reward signal
        self.perf = TierPerformanceTracker(n_tiers)
        self._load_weights()

    # ── weight / velocity initialisation ─────────────────────────────
    def _init_weights(self):
        s1 = np.sqrt(2.0 / self.INPUT_DIM)
        s2 = np.sqrt(2.0 / self.HIDDEN_DIM)
        self.W1 = np.random.randn(self.INPUT_DIM,  self.HIDDEN_DIM) * s1
        self.b1 = np.zeros(self.HIDDEN_DIM)
        self.W2 = np.random.randn(self.HIDDEN_DIM, self.OUTPUT_DIM) * s2
        self.b2 = np.zeros(self.OUTPUT_DIM)

    def _init_velocity(self):
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)

    # ── activations ───────────────────────────────────────────────────
    @staticmethod
    def _relu(x):      return np.maximum(0, x)
    @staticmethod
    def _relu_grad(x): return (x > 0).astype(float)
    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max()); return e / e.sum()

    # ── forward / backward ────────────────────────────────────────────
    def _forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._softmax(z2), (x, z1, a1)

    def _backward(self, probs, label, cache):
        x, z1, a1 = cache
        oh = np.zeros(self.OUTPUT_DIM); oh[label] = 1.0
        dz2 = probs - oh
        dW2 = np.outer(a1, dz2); db2 = dz2.copy()
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_grad(z1)
        dW1 = np.outer(x,  dz1); db1 = dz1.copy()
        return dW1, db1, dW2, db2

    def _apply_grads(self, dW1, db1, dW2, db2):
        self.vW1 = self.MOMENTUM * self.vW1 + self.LEARNING_RATE * dW1
        self.vb1 = self.MOMENTUM * self.vb1 + self.LEARNING_RATE * db1
        self.vW2 = self.MOMENTUM * self.vW2 + self.LEARNING_RATE * dW2
        self.vb2 = self.MOMENTUM * self.vb2 + self.LEARNING_RATE * db2
        self.W1 -= self.vW1; self.b1 -= self.vb1
        self.W2 -= self.vW2; self.b2 -= self.vb2

    # ── Welford online normalisation ──────────────────────────────────
    def _update_norm(self, x: np.ndarray):
        self.feat_n += 1
        d = x - self.feat_mean
        self.feat_mean += d / self.feat_n
        self.feat_var  += d * (x - self.feat_mean)

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.feat_var / max(self.feat_n, 1)) + 1e-8
        return (x - self.feat_mean) / std

    # ── label generation ──────────────────────────────────────────────
    def _rule_label(self, raw: np.ndarray, level: int) -> int:
        fps, cpu, ram = raw[4], raw[0], raw[1]
        pressure = cpu > self.CPU_HEAVY or ram > self.RAM_HEAVY
        if pressure or fps < self.DOWNGRADE_FPS:
            return max(0, level - 1)
        if fps > self.UPGRADE_FPS and not pressure:
            return min(self.n_tiers - 1, level + 1)
        return level

    def _blended_label(self, raw: np.ndarray, level: int,
                       fps: float, avg_conf: float, num_det: int) -> int:
        """
        Three-phase label generation:
          • warm-up  : rule only
          • transition: stochastic blend, weight grows with sample_count
          • mature   : up to MAX_PERF_WEIGHT from performance tracker
        """
        rule = self._rule_label(raw, level)
        # Always record this observation in the performance tracker
        self.perf.update(level, fps, avg_conf, num_det)

        if self.sample_count < self.MIN_SAMPLES:
            return rule                     # phase 1 — warm-up

        perf_best = self.perf.best_adjacent_tier(level)
        if perf_best is None:
            return rule                     # performance says "stay put"

        perf_w = min(self.sample_count / self.PERF_RAMP_SAMPLES,
                     self.MAX_PERF_WEIGHT)
        return perf_best if np.random.random() < perf_w else rule

    # ── observe + train ───────────────────────────────────────────────
    def observe(self, raw: np.ndarray, level: int,
                fps: float, avg_conf: float, num_det: int):
        """Called every frame with fresh telemetry and detection outcomes."""
        self._update_norm(raw)
        label = self._blended_label(raw, level, fps, avg_conf, num_det)
        self.replay_buffer.append((self._normalise(raw).copy(), label))
        self.sample_count  += 1
        self.update_ticker += 1
        if (self.sample_count >= self.MIN_SAMPLES
                and self.update_ticker >= self.UPDATE_INTERVAL
                and len(self.replay_buffer) >= self.BATCH_SIZE):
            self._train_batch()
            self.update_ticker = 0

    def _train_batch(self):
        idx = np.random.choice(len(self.replay_buffer),
                               self.BATCH_SIZE, replace=False)
        aW1 = np.zeros_like(self.W1); ab1 = np.zeros_like(self.b1)
        aW2 = np.zeros_like(self.W2); ab2 = np.zeros_like(self.b2)
        for i in idx:
            f, lbl = self.replay_buffer[i]
            p, cache = self._forward(f)
            dW1, db1, dW2, db2 = self._backward(p, lbl, cache)
            aW1 += dW1; ab1 += db1; aW2 += dW2; ab2 += db2
        s = 1.0 / self.BATCH_SIZE
        self._apply_grads(aW1*s, ab1*s, aW2*s, ab2*s)

    # ── predict ───────────────────────────────────────────────────────
    def predict(self, raw: np.ndarray) -> int:
        if self.sample_count < self.MIN_SAMPLES:
            return int(round(raw[7]))       # warm-up: hold current level
        p, _ = self._forward(self._normalise(raw))
        return int(np.argmax(p))

    def confidence(self, raw: np.ndarray) -> np.ndarray:
        if self.sample_count < self.MIN_SAMPLES:
            p = np.zeros(self.OUTPUT_DIM); p[int(round(raw[7]))] = 1.0
            return p
        p, _ = self._forward(self._normalise(raw))
        return p

    # ── persistence ───────────────────────────────────────────────────
    def save_weights(self):
        data = dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                    feat_mean=self.feat_mean, feat_var=self.feat_var,
                    feat_n=self.feat_n, sample_count=self.sample_count,
                    perf_ema=self.perf.ema, perf_count=self.perf.count)
        with open(self.WEIGHTS_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"[Selector] Saved → {self.WEIGHTS_PATH}")

    def _load_weights(self):
        if not os.path.exists(self.WEIGHTS_PATH): return
        try:
            with open(self.WEIGHTS_PATH, "rb") as f:
                d = pickle.load(f)
            self.W1 = d["W1"]; self.b1 = d["b1"]
            self.W2 = d["W2"]; self.b2 = d["b2"]
            self.feat_mean    = d["feat_mean"]
            self.feat_var     = d["feat_var"]
            self.feat_n       = d["feat_n"]
            self.sample_count = d["sample_count"]
            if "perf_ema" in d:
                self.perf.ema   = d["perf_ema"]
                self.perf.count = d["perf_count"]
            self._init_velocity()
            print(f"[Selector] Loaded weights ({self.sample_count} prior samples)")
        except Exception as e:
            print(f"[Selector] Could not load weights: {e}. Starting fresh.")


# ─────────────────────────────────────────────
# Telemetry helpers
# ─────────────────────────────────────────────
def get_gpu_stats():
    if not GPU_AVAILABLE: return 0.0, 0.0
    try:
        gpus = GPUtil.getGPUs()
        if gpus: return gpus[0].load * 100, gpus[0].memoryUtil * 100
    except Exception: pass
    return 0.0, 0.0


def frame_complexity(frame, scale: float = 0.25) -> float:
    """Tanh-normalised Laplacian variance — proxy for scene edge density."""
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap   = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.tanh(lap / 500.0))


def extract_detection_stats(results):
    """
    Returns (num_det, avg_conf, max_conf, track_count).
    track_count = detections with confidence > 0.5  (reliable / high-quality).
    """
    boxes = results[0].boxes
    n = len(boxes)
    if n == 0:
        return 0, 0.0, 0.0, 0
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    avg_conf    = float(confs.mean())
    max_conf    = float(confs.max())
    track_count = int((confs >= 0.5).sum())
    return n, avg_conf, max_conf, track_count


def compute_motion(prev_gray, curr_frame, scale: float = 0.25):
    """
    Returns (motion_score, curr_gray_small).
    motion_score: normalised mean absolute diff between consecutive frames [0,1].
    """
    small     = cv2.resize(curr_frame, None, fx=scale, fy=scale)
    curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0.0, curr_gray
    diff   = cv2.absdiff(prev_gray, curr_gray)
    motion = float(diff.mean()) / 255.0
    return motion, curr_gray


# ─────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────
def draw_hud(frame, level_names, current_level, fps, probs,
             sample_count, min_samples,
             cpu, ram, gpu,
             num_det, avg_conf, motion,
             perf_ema, perf_counts,
             frames_held, min_hold_frames,
             time_held, min_hold_time):

    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_w, panel_h = 345, 282
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def txt(text, y, color=(210, 210, 210), scale=0.52, thickness=1):
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    mode = "WARM-UP (rule)" if sample_count < min_samples else "ML + perf"
    txt(f"Model: YOLO-{level_names[current_level]}   FPS: {fps:5.1f}",
        30, color=(0, 230, 100), scale=0.65, thickness=2)
    txt(f"Selector: {mode}  [{sample_count} samples]", 52)
    txt(f"CPU {cpu:4.1f}%  RAM {ram:4.1f}%  GPU {gpu:4.1f}%", 72)
    txt(f"Det: {num_det:2d}  AvgConf: {avg_conf:.2f}  Motion: {motion:.3f}", 92)

    # ── per-tier: ML-confidence bar + performance EMA tick ────────────
    #  [ ML confidence bar ████░░░░ ] ← yellow tick = perf EMA
    bar_x, bar_y0 = 16, 112
    bar_max, bar_h = 162, 13
    for i, name in enumerate(level_names):
        y  = bar_y0 + i * 30
        p  = float(probs[i])
        active = (i == current_level)
        bar_color = (0, 210, 80) if active else (90, 90, 90)
        # ML confidence fill
        cv2.rectangle(frame, (bar_x, y),
                      (bar_x + int(bar_max * p), y + bar_h), bar_color, -1)
        # bar outline
        cv2.rectangle(frame, (bar_x, y),
                      (bar_x + bar_max, y + bar_h), (160, 160, 160), 1)
        # performance EMA tick (yellow vertical line)
        ema_x = bar_x + int(bar_max * float(np.clip(perf_ema[i], 0.0, 1.0)))
        cv2.line(frame, (ema_x, y), (ema_x, y + bar_h), (0, 215, 255), 2)
        obs = perf_counts[i]
        txt(f"{name}: {p*100:4.1f}%  ema:{perf_ema[i]:.2f}({obs}obs)",
            y + bar_h, color=(255, 255, 255), scale=0.41)

    # ── switch hold progress bar ──────────────────────────────────────
    hold_y   = bar_y0 + len(level_names) * 30 + 12
    hold_pct = min(frames_held / max(min_hold_frames, 1), 1.0)
    ready    = hold_pct >= 1.0 and time_held >= min_hold_time
    hc       = (0, 255, 80) if ready else (0, 180, 180)
    cv2.rectangle(frame, (bar_x, hold_y),
                  (bar_x + int(bar_max * hold_pct), hold_y + 10), hc, -1)
    cv2.rectangle(frame, (bar_x, hold_y),
                  (bar_x + bar_max, hold_y + 10), (160, 160, 160), 1)
    status = "READY" if ready else "HOLDING"
    txt(f"Switch hold [{status}]: {frames_held}/{min_hold_frames}f  "
        f"{time_held:.1f}/{min_hold_time:.0f}s",
        hold_y + 10, scale=0.41, color=(200, 200, 200))

    txt("'q' quit  |  's' save weights", h - 12,
        color=(120, 120, 120), scale=0.40)
    return frame


# ─────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────
def open_camera():
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release(); continue
        ok = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and frame.max() > 10:
                ok = True; break
        if ok:
            print(f"[Camera] index {idx}  "
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            return cap
        cap.release()
    return None


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main():
    # ── load YOLO models ──────────────────────────────────────────────
    models = {
        "n": YOLO("models/yolo/yolo26n.pt"),
        "s": YOLO("models/yolo/yolo26s.pt"),
        "m": YOLO("models/yolo/yolo26m.pt"),
    }
    model_levels  = ["n", "s", "m"]
    current_level = 0
    current_model = models[model_levels[current_level]]

    selector = AdaptiveModelSelector(n_tiers=len(model_levels))

    # ── hysteresis guard ──────────────────────────────────────────────
    MIN_HOLD_FRAMES = 60        # frames before another switch is allowed
    MIN_HOLD_TIME   = 2.0       # seconds  (both must pass)
    MIN_SWITCH_CONF = 0.55      # ML must be ≥55% confident to trigger a switch

    last_switch_time  = time.time() - MIN_HOLD_TIME   # allow switch at t=0
    last_switch_frame = -MIN_HOLD_FRAMES
    frame_count       = 0

    # ── rolling FPS ───────────────────────────────────────────────────
    fps_window = collections.deque(maxlen=10)
    fps_smooth = 0.0

    # ── temporal state ────────────────────────────────────────────────
    det_ema   = 0.0             # track_age proxy  (EMA of num_detections)
    DET_EMA_A = 0.15
    prev_gray = None            # for motion score

    # ── camera ────────────────────────────────────────────────────────
    cap = open_camera()
    if cap is None:
        print("Error: no webcam found (tried indices 0–3).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WIN = "Adaptive YOLO — ML Selector"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    print("Adaptive YOLO with ML selector — 'q' quit  |  's' save weights")
    consecutive_bad = 0

    while True:
        ret, frame = cap.read()

        # ── blank / corrupt frame guard ───────────────────────────────
        if not ret or frame is None or frame.size == 0 or frame.max() <= 10:
            consecutive_bad += 1
            if consecutive_bad > 30:
                print("Camera stopped delivering frames. Exiting.")
                break
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera...",
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 200, 200), 2, cv2.LINE_AA)
            cv2.imshow(WIN, placeholder)
            if cv2.waitKey(30) & 0xFF == ord("q"): break
            continue
        consecutive_bad = 0
        frame_count += 1

        # ── inference ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = current_model(frame, verbose=False)
        t1 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        fps_window.append(1000.0 / inf_ms if inf_ms > 0 else 0.0)
        fps_smooth = float(np.mean(fps_window))

        # ── detection-quality features ────────────────────────────────
        num_det, avg_conf, max_conf, track_count = extract_detection_stats(results)
        det_ema = (1 - DET_EMA_A) * det_ema + DET_EMA_A * num_det  # track_age

        # ── motion score ──────────────────────────────────────────────
        motion, prev_gray = compute_motion(prev_gray, frame)

        # ── system telemetry ──────────────────────────────────────────
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent
        gpu_pct, gpu_mem_pct = get_gpu_stats()
        complexity = frame_complexity(frame)

        # ── feature vector (14 dims) ──────────────────────────────────
        raw_feat = np.array([
            cpu_pct, ram_pct, gpu_pct, gpu_mem_pct,    # [0–3]  system
            fps_smooth, inf_ms, complexity,             # [4–6]  perf
            float(current_level),                       # [7]    state
            float(num_det), avg_conf, max_conf,         # [8–10] detections
            float(track_count),                         # [11]   track_count
            det_ema,                                    # [12]   track_age
            motion,                                     # [13]   motion
        ], dtype=np.float32)

        # ── ML predict + observe ──────────────────────────────────────
        recommended = selector.predict(raw_feat)
        probs       = selector.confidence(raw_feat)
        selector.observe(raw_feat, current_level,
                         fps_smooth, avg_conf, num_det)

        # ── hysteresis: only switch when all three conditions hold ─────
        now         = time.time()
        frames_held = frame_count - last_switch_frame
        time_held   = now - last_switch_time
        can_switch  = (frames_held >= MIN_HOLD_FRAMES and
                       time_held   >= MIN_HOLD_TIME)

        if (recommended != current_level
                and can_switch
                and probs[recommended] >= MIN_SWITCH_CONF):
            direction = "⬆" if recommended > current_level else "⬇"
            current_level = recommended
            current_model = models[model_levels[current_level]]
            last_switch_time  = now
            last_switch_frame = frame_count
            print(f"{direction} ML → YOLO-{model_levels[current_level]}"
                  f"  conf={probs[recommended]:.0%}"
                  f"  held={frames_held}f/{time_held:.1f}s"
                  f"  FPS={fps_smooth:.1f}"
                  f"  det={num_det} avgConf={avg_conf:.2f}"
                  f"  motion={motion:.3f}"
                  f"  samples={selector.sample_count}")

        # ── draw ──────────────────────────────────────────────────────
        annotated = results[0].plot()
        if annotated is None or annotated.size == 0:
            annotated = frame.copy()

        dh, dw = annotated.shape[:2]
        if dw != 1280:
            annotated = cv2.resize(annotated,
                                   (1280, int(dh * 1280 / dw)),
                                   interpolation=cv2.INTER_LINEAR)

        annotated = draw_hud(
            annotated, model_levels, current_level, fps_smooth, probs,
            selector.sample_count, selector.MIN_SAMPLES,
            cpu_pct, ram_pct, gpu_pct,
            num_det, avg_conf, motion,
            selector.perf.ema, selector.perf.count,
            frames_held, MIN_HOLD_FRAMES,
            time_held, MIN_HOLD_TIME,
        )
        cv2.imshow(WIN, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("s"): selector.save_weights()

    selector.save_weights()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()