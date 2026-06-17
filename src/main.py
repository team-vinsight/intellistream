"""
main.py
-------
Entry point for the Adaptive YOLO system.

Orchestrates the main capture-infer-select-render loop:

  1. Load configuration from config.yaml
  2. Load YOLO model variants and resolve configured class IDs
  3. Initialise the PyTorch-backed adaptive model selector
  4. Open the camera or video file
  5. Per-frame loop:
       a. Read frame
       b. Run YOLO inference
       c. Extract telemetry and detection statistics
       d. Build the 14-dim feature vector
       e. Predict recommended tier; observe for online training
       f. Apply hysteresis guard; switch tier if conditions are met
       g. Render masked output frame + HUD overlay
       h. Collect metrics
  6. On exit: save selector weights, flush metrics CSV, generate graphs

Run with:
    cd <project_root>

    # Live webcam (default -- auto-probes indices 0-3)
    python src/main.py

    # Specific webcam index
    python src/main.py --source 1

    # Local video file
    python src/main.py --source path/to/video.mp4

    # Custom config
    python src/main.py --config path/to/custom_config.yaml

    # Combined
    python src/main.py --source videos/clip.mp4 --config src/config.yaml
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

# -- Local modules (all inside src/) ------------------------------------------
from camera        import open_camera, is_valid_frame
from config        import load_config
from detector      import load_models, resolve_class_ids, run_inference, render_masked_frame
from metrics       import MetricsCollector
from selector      import AdaptiveModelSelector
from telemetry     import get_cpu_ram, get_gpu_stats, frame_complexity, compute_motion, extract_detection_stats
from utils         import build_feature_vector, placeholder_frame
from visualization import draw_hud


# -----------------------------------------------------------------------------
# Source helpers
# -----------------------------------------------------------------------------

def _open_source(source: str, cfg) -> tuple:
    """
    Open a video capture from either a webcam index or a video file path.

    Args:
        source: A digit string (webcam index, e.g. "0") or a path to a
                local video file (e.g. "videos/clip.mp4").
        cfg:    Application config (used for camera resolution when webcam).

    Returns:
        (cap, is_file) -- the opened VideoCapture and a flag indicating
        whether the source is a file (True) or a live webcam (False).
        Returns (None, False) on failure.
    """
    # Webcam index
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"[Main] Error: could not open webcam index {idx}.")
            return None, False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Main] Opened webcam index {idx}  {actual_w}x{actual_h}")
        return cap, False

    # Video file path
    if not os.path.isfile(source):
        print(f"[Main] Error: video file not found: '{source}'")
        return None, False

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Main] Error: could not open video file '{source}'.")
        return None, False

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    w_in   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[Main] Opened video file '{source}'  "
        f"{w_in}x{h_in}  {fps_in:.1f} fps  {total} frames"
    )
    return cap, True


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main(config_path=None, source="0"):
    """
    Run the adaptive YOLO detection loop.

    Args:
        config_path: Optional path to a custom config.yaml.  Defaults to
                     src/config.yaml when None.
        source:      Video source -- a webcam index string (e.g. "0") or
                     a path to a local video file (e.g. "videos/clip.mp4").
    """
    # -- Configuration ---------------------------------------------------------
    cfg = load_config(config_path)

    # -- YOLO models -----------------------------------------------------------
    models        = load_models(cfg)
    model_levels  = cfg.model.tiers
    current_level = 0
    current_model = models[model_levels[current_level]]

    class_ids = resolve_class_ids(current_model, cfg.detection.classes)
    if class_ids:
        print(f"[Main] Filtering to classes: {cfg.detection.classes} -> IDs {class_ids}")
    else:
        print("[Main] No class filter -- detecting all classes.")

    # -- Adaptive selector (PyTorch MLP) ---------------------------------------
    selector = AdaptiveModelSelector(cfg.selector, n_tiers=len(model_levels))

    # -- Hysteresis guard ------------------------------------------------------
    hyst              = cfg.hysteresis
    last_switch_time  = time.time() - hyst.min_hold_time
    last_switch_frame = -hyst.min_hold_frames
    frame_count       = 0

    # -- Rolling FPS -----------------------------------------------------------
    fps_window = collections.deque(maxlen=10)
    fps_smooth = 0.0

    # -- Temporal state --------------------------------------------------------
    det_ema   = 0.0
    DET_EMA_A = 0.15
    prev_gray = None

    # -- Metrics collector -----------------------------------------------------
    metrics = MetricsCollector(cfg.metrics)

    # -- Video source ----------------------------------------------------------
    cap, is_file = _open_source(source, cfg)
    if cap is None:
        if source == "0":
            print("[Main] Auto-probing webcam indices 0-3 ...")
            cap = open_camera(cfg.camera)
        if cap is None:
            print("[Main] Error: no valid video source found. Exiting.")
            sys.exit(1)
        is_file = False

    WIN = "Adaptive YOLO -- ML Selector"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    display_w = cfg.camera.width
    display_h = cfg.camera.height
    if is_file:
        display_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        display_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.resizeWindow(WIN, display_w, display_h)

    source_label = f"file: {os.path.basename(source)}" if is_file else f"webcam:{source}"
    print(
        f"[Main] Running ({source_label}) -- 'q' quit  |  's' save weights\n"
        f"       Output: masked frame (black rectangles over detected objects)\n"
        f"       Reports will be saved to '{cfg.metrics.reports_dir}/' on exit."
    )

    consecutive_bad = 0

    # -------------------------------------------------------------------------
    # Frame loop
    # -------------------------------------------------------------------------
    while True:
        ret, frame = cap.read()

        # Video file: loop back to start when the file ends
        if is_file and not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        # -- Blank / corrupt frame guard ---------------------------------------
        if not ret or not is_valid_frame(frame):
            consecutive_bad += 1
            if consecutive_bad > cfg.camera.max_bad_frames:
                print("[Main] Source stopped delivering frames. Exiting.")
                break
            cv2.imshow(WIN, placeholder_frame(display_w, display_h))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        consecutive_bad = 0
        frame_count    += 1

        # -- Inference ---------------------------------------------------------
        t0      = time.perf_counter()
        results = run_inference(current_model, frame)
        t1      = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        fps_window.append(1000.0 / inf_ms if inf_ms > 0 else 0.0)
        fps_smooth = float(np.mean(fps_window))

        # -- Detection statistics ----------------------------------------------
        num_det, avg_conf, max_conf, track_count = extract_detection_stats(
            results, class_ids=class_ids if class_ids else None
        )
        det_ema = (1 - DET_EMA_A) * det_ema + DET_EMA_A * num_det

        # -- Motion score ------------------------------------------------------
        motion, prev_gray = compute_motion(prev_gray, frame)

        # -- System telemetry --------------------------------------------------
        cpu_pct, ram_pct     = get_cpu_ram()
        gpu_pct, gpu_mem_pct = get_gpu_stats()
        complexity           = frame_complexity(frame)

        # -- 14-dim feature vector ---------------------------------------------
        raw_feat = build_feature_vector(
            cpu_pct, ram_pct, gpu_pct, gpu_mem_pct,
            fps_smooth, inf_ms, complexity,
            current_level,
            num_det, avg_conf, max_conf,
            track_count, det_ema, motion,
        )

        # -- ML predict + observe ----------------------------------------------
        recommended = selector.predict(raw_feat)
        probs       = selector.confidence(raw_feat)
        selector.observe(raw_feat, current_level, fps_smooth, avg_conf, num_det)

        # -- Hysteresis --------------------------------------------------------
        now         = time.time()
        frames_held = frame_count - last_switch_frame
        time_held   = now - last_switch_time
        can_switch  = (
            frames_held >= hyst.min_hold_frames
            and time_held >= hyst.min_hold_time
        )

        if (
            recommended != current_level
            and can_switch
            and probs[recommended] >= hyst.min_switch_conf
        ):
            direction     = "^" if recommended > current_level else "v"
            current_level = recommended
            current_model = models[model_levels[current_level]]
            last_switch_time  = now
            last_switch_frame = frame_count
            print(
                f"{direction} ML -> YOLO-{model_levels[current_level]}"
                f"  conf={probs[recommended]:.0%}"
                f"  held={frames_held}f/{time_held:.1f}s"
                f"  FPS={fps_smooth:.1f}"
                f"  det={num_det} avgConf={avg_conf:.2f}"
                f"  motion={motion:.3f}"
                f"  samples={selector.sample_count}"
            )

        # -- Metrics collection ------------------------------------------------
        metrics.record(
            fps            = fps_smooth,
            inference_ms   = inf_ms,
            cpu_pct        = cpu_pct,
            ram_pct        = ram_pct,
            gpu_pct        = gpu_pct,
            num_detections = num_det,
            model_tier     = current_level,
        )

        # -- Render: masked frame + HUD ----------------------------------------
        output = render_masked_frame(
            frame, results, class_ids,
            conf_threshold=cfg.detection.confidence_threshold,
            target_width=display_w,
        )
        output = draw_hud(
            output, model_levels, current_level, fps_smooth, probs,
            selector.sample_count, selector.MIN_SAMPLES,
            cpu_pct, ram_pct, gpu_pct,
            num_det, avg_conf, motion,
            selector.perf.ema, selector.perf.count,
            frames_held, hyst.min_hold_frames,
            time_held, hyst.min_hold_time,
        )

        cv2.imshow(WIN, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            selector.save_weights()

    # -- Teardown --------------------------------------------------------------
    print("[Main] Shutting down ...")
    selector.save_weights()
    cap.release()
    cv2.destroyAllWindows()

    metrics.save()
    metrics.generate_reports()
    print("[Main] Done.")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive YOLO with PyTorch MLP selector and object masking.\n\n"
            "SOURCE can be a webcam index (e.g. 0, 1) or a path to a local\n"
            "video file (e.g. videos/clip.mp4)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        metavar="SOURCE",
        default="0",
        help=(
            "Video source: webcam index (default: 0) or path to a video file. "
            "Examples: --source 1  |  --source videos/clip.mp4"
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a custom config.yaml (default: src/config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config, source=args.source)
