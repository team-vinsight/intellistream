"""
telemetry.py
────────────
System-level telemetry helpers: CPU, RAM, GPU usage, frame complexity,
motion score, and detection statistics.

All functions are pure (no side effects) and return plain Python scalars
or NumPy arrays so they can be used from any module without coupling.
"""

from __future__ import annotations

from typing import Tuple, Optional

import cv2
import numpy as np
import psutil

# ── Optional GPU monitoring ───────────────────────────────────────────────────
try:
    import GPUtil
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# System resource telemetry
# ─────────────────────────────────────────────────────────────────────────────

def get_cpu_ram() -> Tuple[float, float]:
    """
    Return current CPU and RAM utilisation as percentages.

    Uses ``psutil`` with ``interval=None`` (non-blocking, returns the value
    from the last call) to avoid stalling the main loop.
    """
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    return cpu, ram


def get_gpu_stats() -> Tuple[float, float]:
    """
    Return (gpu_load_pct, gpu_mem_pct) for the first available GPU.

    Returns (0.0, 0.0) when GPUtil is not installed or no GPU is found.
    """
    if not _GPU_AVAILABLE:
        return 0.0, 0.0
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100.0, gpus[0].memoryUtil * 100.0
    except Exception:
        pass
    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level features
# ─────────────────────────────────────────────────────────────────────────────

def frame_complexity(frame: np.ndarray, scale: float = 0.25) -> float:
    """
    Compute a tanh-normalised Laplacian variance as a proxy for scene
    edge density (higher → more texture / detail).

    The frame is downscaled before computing the Laplacian to reduce noise
    and speed up the calculation.

    Args:
        frame: BGR image (H × W × 3, uint8).
        scale: Downscale factor applied before the Laplacian.

    Returns:
        Scalar in (0, 1).
    """
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap   = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.tanh(lap / 500.0))


def compute_motion(
    prev_gray: Optional[np.ndarray],
    curr_frame: np.ndarray,
    scale: float = 0.25,
) -> Tuple[float, np.ndarray]:
    """
    Estimate inter-frame motion as a normalised mean absolute difference.

    Args:
        prev_gray:  Downscaled grayscale of the *previous* frame, or None
                    on the very first call.
        curr_frame: Current BGR frame.
        scale:      Downscale factor applied before differencing.

    Returns:
        (motion_score, curr_gray_small)
        motion_score: float in [0, 1] — 0 means no motion.
        curr_gray_small: the downscaled grayscale to pass as prev_gray next call.
    """
    small     = cv2.resize(curr_frame, None, fx=scale, fy=scale)
    curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0.0, curr_gray
    diff   = cv2.absdiff(prev_gray, curr_gray)
    motion = float(diff.mean()) / 255.0
    return motion, curr_gray


# ─────────────────────────────────────────────────────────────────────────────
# Detection statistics
# ─────────────────────────────────────────────────────────────────────────────

def extract_detection_stats(
    results,
    class_ids: Optional[list] = None,
) -> Tuple[int, float, float, int]:
    """
    Compute aggregate statistics from YOLO results, optionally filtered to
    a specific set of class IDs.

    Args:
        results:   List of Ultralytics ``Results`` objects (one per image).
        class_ids: If provided, only boxes whose class ID is in this list
                   are counted.  Pass ``None`` to include all classes.

    Returns:
        (num_det, avg_conf, max_conf, track_count)
        num_det:     Number of detections (after class filtering).
        avg_conf:    Mean confidence of those detections (0.0 if none).
        max_conf:    Maximum confidence (0.0 if none).
        track_count: Detections with confidence ≥ 0.5 (reliable objects).
    """
    boxes = results[0].boxes
    if len(boxes) == 0:
        return 0, 0.0, 0.0, 0

    confs = (
        boxes.conf.cpu().numpy()
        if hasattr(boxes.conf, "cpu")
        else np.array(boxes.conf)
    )
    cls_ids = (
        boxes.cls.cpu().numpy().astype(int)
        if hasattr(boxes.cls, "cpu")
        else np.array(boxes.cls, dtype=int)
    )

    # Apply class filter
    if class_ids is not None and len(class_ids) > 0:
        mask  = np.isin(cls_ids, class_ids)
        confs = confs[mask]

    n = len(confs)
    if n == 0:
        return 0, 0.0, 0.0, 0

    avg_conf    = float(confs.mean())
    max_conf    = float(confs.max())
    track_count = int((confs >= 0.5).sum())
    return n, avg_conf, max_conf, track_count
