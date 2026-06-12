"""
utils.py
────────
Shared utility helpers that do not belong to any single module.

Currently provides:
  • build_feature_vector — assembles the 14-dimensional input for the MLP
  • placeholder_frame    — generates a "Waiting for camera…" frame
"""

from __future__ import annotations

import cv2
import numpy as np


def build_feature_vector(
    cpu_pct:       float,
    ram_pct:       float,
    gpu_pct:       float,
    gpu_mem_pct:   float,
    fps_smooth:    float,
    inf_ms:        float,
    complexity:    float,
    current_level: int,
    num_det:       int,
    avg_conf:      float,
    max_conf:      float,
    track_count:   int,
    det_ema:       float,
    motion:        float,
) -> np.ndarray:
    """
    Assemble the 14-dimensional feature vector consumed by the MLP selector.

    Feature layout
    ──────────────
      [0]  cpu_pct          CPU utilisation (%)
      [1]  ram_pct          RAM utilisation (%)
      [2]  gpu_pct          GPU load (%)
      [3]  gpu_mem_pct      GPU memory utilisation (%)
      [4]  fps_smooth       10-frame rolling mean FPS
      [5]  inf_ms           Raw inference latency (ms)
      [6]  complexity       Tanh-normalised Laplacian variance
      [7]  current_level    Active tier index (0 / 1 / 2)
      [8]  num_det          Filtered detection count
      [9]  avg_conf         Mean confidence of filtered detections
      [10] max_conf         Maximum confidence of filtered detections
      [11] track_count      Detections with confidence ≥ 0.5
      [12] det_ema          EMA of num_det (temporal stability proxy)
      [13] motion           Normalised inter-frame motion score

    Returns:
        float32 array of shape (14,).
    """
    return np.array(
        [
            cpu_pct, ram_pct, gpu_pct, gpu_mem_pct,    # [0–3]  system
            fps_smooth, inf_ms, complexity,             # [4–6]  performance
            float(current_level),                       # [7]    state
            float(num_det), avg_conf, max_conf,         # [8–10] detections
            float(track_count),                         # [11]   track_count
            det_ema,                                    # [12]   track_age
            motion,                                     # [13]   motion
        ],
        dtype=np.float32,
    )


def placeholder_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Return a black frame with a "Waiting for camera…" message.

    Used in the main loop when the camera delivers a blank or corrupt frame.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Waiting for camera...",
        (width // 4, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return frame
