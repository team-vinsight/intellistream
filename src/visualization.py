"""
visualization.py
────────────────
HUD (heads-up display) overlay drawn on top of every output frame.

The HUD shows:
  • Active model tier and current FPS
  • Selector mode (warm-up / ML + perf) and sample count
  • System resource bars (CPU / RAM / GPU)
  • Detection count, average confidence, and motion score
  • Per-tier ML-confidence bars with performance-EMA tick marks
  • Switch-hold progress bar
  • Keyboard shortcut reminder
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_hud(
    frame: np.ndarray,
    level_names: list,
    current_level: int,
    fps: float,
    probs: np.ndarray,
    sample_count: int,
    min_samples: int,
    cpu: float,
    ram: float,
    gpu: float,
    num_det: int,
    avg_conf: float,
    motion: float,
    perf_ema: list,
    perf_counts: list,
    frames_held: int,
    min_hold_frames: int,
    time_held: float,
    min_hold_time: float,
) -> np.ndarray:
    """
    Render the HUD overlay onto *frame* in-place and return it.

    Args:
        frame:           BGR output frame (already masked).
        level_names:     Ordered list of tier labels, e.g. ["n", "s", "m"].
        current_level:   Index of the currently active tier.
        fps:             Smoothed FPS (10-frame rolling mean).
        probs:           MLP softmax probability vector (one value per tier).
        sample_count:    Total observations seen by the selector.
        min_samples:     Warm-up threshold from selector config.
        cpu:             CPU utilisation (%).
        ram:             RAM utilisation (%).
        gpu:             GPU utilisation (%).
        num_det:         Filtered detection count for this frame.
        avg_conf:        Mean confidence of filtered detections.
        motion:          Normalised inter-frame motion score [0, 1].
        perf_ema:        Per-tier EMA performance scores (list of floats).
        perf_counts:     Per-tier observation counts (list of ints).
        frames_held:     Frames elapsed since the last tier switch.
        min_hold_frames: Minimum frames required before another switch.
        time_held:       Seconds elapsed since the last tier switch.
        min_hold_time:   Minimum seconds required before another switch.

    Returns:
        The same *frame* array with the HUD drawn on it.
    """
    h, w = frame.shape[:2]

    # ── Semi-transparent dark panel background ────────────────────────────────
    overlay   = frame.copy()
    panel_w   = 345
    panel_h   = 282
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Text helper ───────────────────────────────────────────────────────────
    def txt(
        text: str,
        y: int,
        color: tuple = (210, 210, 210),
        scale: float = 0.52,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            frame, text, (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
        )

    # ── Header rows ───────────────────────────────────────────────────────────
    mode = "WARM-UP (rule)" if sample_count < min_samples else "ML + perf"
    txt(
        f"Model: YOLO-{level_names[current_level]}   FPS: {fps:5.1f}",
        30, color=(0, 230, 100), scale=0.65, thickness=2,
    )
    txt(f"Selector: {mode}  [{sample_count} samples]", 52)
    txt(f"CPU {cpu:4.1f}%  RAM {ram:4.1f}%  GPU {gpu:4.1f}%", 72)
    txt(f"Det: {num_det:2d}  AvgConf: {avg_conf:.2f}  Motion: {motion:.3f}", 92)

    # ── Per-tier confidence bars ───────────────────────────────────────────────
    # Each bar shows:
    #   • Filled portion = MLP softmax probability (green if active, grey otherwise)
    #   • Yellow vertical tick = performance EMA score
    bar_x   = 16
    bar_y0  = 112
    bar_max = 162
    bar_h   = 13

    for i, name in enumerate(level_names):
        y      = bar_y0 + i * 30
        p      = float(probs[i])
        active = (i == current_level)

        bar_color = (0, 210, 80) if active else (90, 90, 90)

        # ML confidence fill
        cv2.rectangle(
            frame,
            (bar_x, y),
            (bar_x + int(bar_max * p), y + bar_h),
            bar_color, -1,
        )
        # Bar outline
        cv2.rectangle(
            frame,
            (bar_x, y),
            (bar_x + bar_max, y + bar_h),
            (160, 160, 160), 1,
        )
        # Performance EMA tick (yellow vertical line)
        ema_x = bar_x + int(bar_max * float(np.clip(perf_ema[i], 0.0, 1.0)))
        cv2.line(frame, (ema_x, y), (ema_x, y + bar_h), (0, 215, 255), 2)

        obs = perf_counts[i]
        txt(
            f"{name}: {p * 100:4.1f}%  ema:{perf_ema[i]:.2f}({obs}obs)",
            y + bar_h,
            color=(255, 255, 255),
            scale=0.41,
        )

    # ── Switch-hold progress bar ──────────────────────────────────────────────
    hold_y   = bar_y0 + len(level_names) * 30 + 12
    hold_pct = min(frames_held / max(min_hold_frames, 1), 1.0)
    ready    = hold_pct >= 1.0 and time_held >= min_hold_time
    hc       = (0, 255, 80) if ready else (0, 180, 180)

    cv2.rectangle(
        frame,
        (bar_x, hold_y),
        (bar_x + int(bar_max * hold_pct), hold_y + 10),
        hc, -1,
    )
    cv2.rectangle(
        frame,
        (bar_x, hold_y),
        (bar_x + bar_max, hold_y + 10),
        (160, 160, 160), 1,
    )
    status = "READY" if ready else "HOLDING"
    txt(
        f"Switch hold [{status}]: {frames_held}/{min_hold_frames}f  "
        f"{time_held:.1f}/{min_hold_time:.0f}s",
        hold_y + 10,
        scale=0.41,
        color=(200, 200, 200),
    )

    # ── Keyboard shortcut reminder ────────────────────────────────────────────
    txt("'q' quit  |  's' save weights", h - 12,
        color=(120, 120, 120), scale=0.40)

    return frame
