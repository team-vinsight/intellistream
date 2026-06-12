"""
camera.py
─────────
Camera initialisation and frame-validity helpers.

Tries camera indices 0–3 in order and returns the first one that delivers
a non-blank frame.  Exposes a simple guard function used in the main loop
to detect corrupt or blank frames.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import CameraConfig


def open_camera(cfg: CameraConfig) -> Optional[cv2.VideoCapture]:
    """
    Probe camera indices 0–3 and return the first working capture device.

    A camera is considered "working" if it delivers at least one frame
    whose maximum pixel value exceeds 10 (i.e. not a pure-black frame).

    Args:
        cfg: Camera configuration (width, height).

    Returns:
        An opened ``cv2.VideoCapture`` object, or ``None`` if no camera
        was found.
    """
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        # Attempt up to 10 reads to confirm the camera delivers real frames
        ok = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and frame.max() > 10:
                ok = True
                break

        if ok:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Camera] Opened index {idx}  {actual_w}×{actual_h}")
            return cap

        cap.release()

    return None


def is_valid_frame(frame: Optional[np.ndarray]) -> bool:
    """
    Return True if *frame* is a non-None, non-empty, non-blank image.

    A frame is considered blank when its maximum pixel value is ≤ 10,
    which typically indicates a camera that has not warmed up yet or a
    hardware error.
    """
    return (
        frame is not None
        and frame.size > 0
        and frame.max() > 10
    )
