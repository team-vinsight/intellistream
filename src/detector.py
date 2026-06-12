"""
detector.py
───────────
YOLO model loading, inference, class-ID resolution, and masked-frame
rendering.

Key responsibilities
────────────────────
* Load all three YOLO model variants at startup.
* Resolve configured class *names* to integer IDs using the model's own
  class map (so the config stays human-readable).
* Run inference and return raw Ultralytics ``Results``.
* Render the output frame: instead of YOLO bounding boxes / labels, draw
  a solid black rectangle over every detected object that belongs to a
  configured class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import AppConfig


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_models(cfg: AppConfig) -> Dict[str, YOLO]:
    """
    Load all YOLO model variants defined in the configuration.

    Returns a dict keyed by tier label (e.g. ``{"n": <YOLO>, "s": <YOLO>,
    "m": <YOLO>}``).
    """
    tier_paths = {
        "n": cfg.model.nano,
        "s": cfg.model.small,
        "m": cfg.model.medium,
    }
    models: Dict[str, YOLO] = {}
    for tier in cfg.model.tiers:
        path = tier_paths[tier]
        print(f"[Detector] Loading YOLO-{tier} from '{path}' …")
        models[tier] = YOLO(path)
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Class-ID resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_class_ids(
    model: YOLO,
    class_names: List[str],
) -> List[int]:
    """
    Convert a list of human-readable class names to integer IDs using the
    model's own ``names`` dictionary.

    Unknown names are warned about and skipped.  An empty ``class_names``
    list returns an empty list, which the rest of the code interprets as
    "detect all classes".

    Args:
        model:        Any loaded YOLO model (all variants share the same
                      COCO class map).
        class_names:  List of strings from ``config.yaml``.

    Returns:
        List of integer class IDs.
    """
    if not class_names:
        return []

    # model.names is {int: str}; invert it for lookup
    name_to_id: Dict[str, int] = {v: k for k, v in model.names.items()}
    ids: List[int] = []
    for name in class_names:
        if name in name_to_id:
            ids.append(name_to_id[name])
        else:
            print(f"[Detector] Warning: class '{name}' not found in model — skipped.")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model: YOLO, frame: np.ndarray):
    """
    Run YOLO inference on *frame* and return the raw Ultralytics Results list.

    ``verbose=False`` suppresses per-frame console output.
    """
    return model(frame, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Masked-frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_masked_frame(
    frame: np.ndarray,
    results,
    class_ids: List[int],
    conf_threshold: float = 0.25,
    target_width: int = 1280,
) -> np.ndarray:
    """
    Produce the output frame with detected objects obscured by solid black
    rectangles.

    No YOLO bounding boxes, labels, or confidence values are drawn.
    Only detections whose class ID is in *class_ids* (and whose confidence
    exceeds *conf_threshold*) are masked.  If *class_ids* is empty, all
    detections are masked.

    Args:
        frame:          Original BGR frame from the camera.
        results:        Ultralytics Results list from ``run_inference``.
        class_ids:      Integer class IDs to mask.  Empty → mask all.
        conf_threshold: Minimum confidence to apply a mask.
        target_width:   Output frame is resized to this width (aspect-ratio
                        preserved).

    Returns:
        BGR frame (H × target_width × 3) with masked regions.
    """
    output = frame.copy()
    boxes  = results[0].boxes

    if len(boxes) > 0:
        confs = (
            boxes.conf.cpu().numpy()
            if hasattr(boxes.conf, "cpu")
            else np.array(boxes.conf)
        )
        cls_arr = (
            boxes.cls.cpu().numpy().astype(int)
            if hasattr(boxes.cls, "cpu")
            else np.array(boxes.cls, dtype=int)
        )
        # xyxy coordinates in the original frame's pixel space
        xyxy = (
            boxes.xyxy.cpu().numpy()
            if hasattr(boxes.xyxy, "cpu")
            else np.array(boxes.xyxy)
        )

        for i in range(len(boxes)):
            conf = float(confs[i])
            cid  = int(cls_arr[i])

            # Skip low-confidence detections
            if conf < conf_threshold:
                continue

            # Skip classes not in the filter list (empty list = all classes)
            if class_ids and cid not in class_ids:
                continue

            x1, y1, x2, y2 = map(int, xyxy[i])
            # Clamp to frame boundaries
            h, w = output.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw solid black mask over the detected region
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    # Resize to target width while preserving aspect ratio
    oh, ow = output.shape[:2]
    if ow != target_width:
        output = cv2.resize(
            output,
            (target_width, int(oh * target_width / ow)),
            interpolation=cv2.INTER_LINEAR,
        )

    return output
