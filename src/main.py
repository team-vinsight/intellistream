"""
main.py
───────
Entry point for the Adaptive YOLO system.

Orchestrates the main capture-infer-select-render loop:

  1. Load configuration from config.yaml
  2. Load YOLO model variants and resolve configured class IDs
  3. Initialise the PyTorch-backed adaptive model selector
  4. Open the camera
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
    python src/main.py
    python src/main.py --config path/to/custom_config.yaml
"""

from __future__ import annotations

import argparse
import collections
import sys
import time

import cv2
import numpy as np

# ── Local modules (all inside src/) ──────────────────────────────────────────
from camera        import open_camera, is_valid_frame
from config        import load_config
from detector      import load_models, resolve_class_ids, run_inference, render_masked_frame
from metrics       import MetricsCollector
from selector      import AdaptiveModelSelector
from telemetry     import get_cpu_ram, get_gpu_stats, frame_complexity, compute_motion, extract_detection_stats
from utils         import build_feature_vector, placeholder_frame
from visualization import draw_hud


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str | None = None) -> None:
    """
    Run the adaptive YOLO detection loop.

    Args:
        config_path: Optional path to a custom config.yaml.  Defaults to
                     ``src/config.yaml`` when None.
    """
    # ── Configuration ─────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # ── YOLO models ───────────────────────────────────────────────────────────
    models       = load_models(cfg)
    model_levels = cfg.model.tiers          # e.g. ["n", "s", "m"]
    current_level = 0
    current_model = models[model_levels[current_level]]

    # Resolve class names → integer IDs using the first (lightest) model's map
    class_ids = resolve_class_ids(current_model, cfg.detection.classes)
    if class_ids:
        print(f"[Main] Filtering to classes: {cfg.detection.classes} → IDs {class_ids}")
    else:
        print("[Main] No class filter — detecting all classes.")

    # ── Adaptive selector (PyTorch MLP) ───────────────────────────────────────
    selector = AdaptiveModelSelector(cfg.selector, n_tiers=len(model_levels))

    # ── Hysteresis guard ──────────────────────────────────────────────────────
    hyst             = cfg.hysteresis
    last_switch_time  = time.time() - hyst.min_hold_time   # allow switch at t=0
    last_switch_frame = -hyst.min_hold_frames
    frame_count       = 0

    # ── Rolling FPS ───────────────────────────────────────────────────────────
    fps_window: collections.deque = collections.deque(maxlen=10)
    fps_smooth  = 0.0

    # ── Temporal state ────────────────────────────────────────────────────────
    det_ema   = 0.0     # EMA of num_detections — used as "track_age" feature
    DET_EMA_A = 0.15
    prev_gray = None    # previous downscaled grayscale frame for motion score

    # ── Metrics collector ─────────────────────────────────────────────────────
    metrics = MetricsCollector(cfg.metrics)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = open_camera(cfg.camera)
    if cap is None:
        print("[Main] Error: no webcam found (tried indices 0–3). Exiting.")
        sys.exit(1)

    WIN = "Adaptive YOLO — ML Selector"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cfg.camera.width, cfg.camera.height)

    print(
        f"[Main] Running — 'q' quit  |  's' save weights\n"
        f"       Output: masked frame (black rectangles over detected objects)\n"
        f"       Reports will be saved to '{cfg.metrics.reports_dir}/' on exit."
    )

    consecutive_bad = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Frame loop
    # ─────────────────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()

        # ── Blank / corrupt frame guard ───────────────────────────────────────
        if not ret or not is_valid_frame(frame):
            consecutive_bad += 1
            if consecutive_bad > cfg.camera.max_bad_frames:
                print("[Main] Camera stopped delivering frames. Exiting.")
                break
            cv2.imshow(WIN, placeholder_frame(cfg.camera.width, cfg.camera.height))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        consecutive_bad = 0
        frame_count    += 1

        # ── Inference ─────────────────────────────────────────────────────────
        t0      = time.perf_counter()
        results = run_inference(current_model, frame)
        t1      = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        fps_window.append(1000.0 / inf_ms if inf_ms > 0 else 0.0)
        fps_smooth = float(np.mean(fps_window))

        # ── Detection statistics (filtered to configured classes) ─────────────
        num_det, avg_conf, max_conf, track_count = extract_detection_stats(
            results, class_ids=class_ids if class_ids else None
        )
        det_ema = (1 - DET_EMA_A) * det_ema + DET_EMA_A * num_det

        # ── Motion score ──────────────────────────────────────────────────────
        motion, prev_gray = compute_motion(prev_gray, frame)

        # ── System telemetry ──────────────────────────────────────────────────
        cpu_pct, ram_pct    = get_cpu_ram()
        gpu_pct, gpu_mem_pct = get_gpu_stats()
        complexity           = frame_complexity(frame)

        # ── 14-dim feature vector ─────────────────────────────────────────────
        raw_feat = build_feature_vector(
            cpu_pct, ram_pct, gpu_pct, gpu_mem_pct,
            fps_smooth, inf_ms, complexity,
            current_level,
            num_det, avg_conf, max_conf,
            track_count, det_ema, motion,
        )

        # ── ML predict + observe ──────────────────────────────────────────────
        recommended = selector.predict(raw_feat)
        probs       = selector.confidence(raw_feat)
        selector.observe(raw_feat, current_level, fps_smooth, avg_conf, num_det)

        # ── Hysteresis: only switch when all three conditions hold ─────────────
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
            direction     = "⬆" if recommended > current_level else "⬇"
            current_level = recommended
            current_model = models[model_levels[current_level]]
            last_switch_time  = now
            last_switch_frame = frame_count
            print(
                f"{direction} ML → YOLO-{model_levels[current_level]}"
                f"  conf={probs[recommended]:.0%}"
                f"  held={frames_held}f/{time_held:.1f}s"
                f"  FPS={fps_smooth:.1f}"
                f"  det={num_det} avgConf={avg_conf:.2f}"
                f"  motion={motion:.3f}"
                f"  samples={selector.sample_count}"
            )

        # ── Metrics collection ────────────────────────────────────────────────
        metrics.record(
            fps            = fps_smooth,
            inference_ms   = inf_ms,
            cpu_pct        = cpu_pct,
            ram_pct        = ram_pct,
            gpu_pct        = gpu_pct,
            num_detections = num_det,
            model_tier     = current_level,
        )

        # ── Render: masked frame + HUD ────────────────────────────────────────
        output = render_masked_frame(
            frame, results, class_ids,
            conf_threshold=cfg.detection.confidence_threshold,
            target_width=cfg.camera.width,
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

    # ── Teardown ──────────────────────────────────────────────────────────────
    print("[Main] Shutting down …")
    selector.save_weights()
    cap.release()
    cv2.destroyAllWindows()

    metrics.save()
    metrics.generate_reports()
    print("[Main] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive YOLO with PyTorch MLP selector and object masking."
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a custom config.yaml (default: src/config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
