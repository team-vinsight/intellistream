"""
benchmark.py
------------
Pipeline performance benchmark for the Adaptive YOLO system.

Compares four video processing pipelines on a local video file:
  1. Adaptive  -- MLP-driven model selector (nano / small / medium)
  2. Fixed-N   -- always uses YOLO nano
  3. Fixed-S   -- always uses YOLO small
  4. Fixed-M   -- always uses YOLO medium

Metrics collected per frame
----------------------------
  fps            -- instantaneous FPS (1000 / inference_ms)
  inference_ms   -- raw YOLO inference latency
  cpu_pct        -- CPU utilisation (%)
  ram_pct        -- RAM utilisation (%)
  gpu_pct        -- GPU load (%)
  num_detections -- detection count (all classes)

Output
------
  reports/benchmark_results.csv   -- raw per-frame data for all pipelines
  reports/benchmark_fps.png
  reports/benchmark_latency.png
  reports/benchmark_resources.png
  reports/benchmark_detections.png
  reports/benchmark_summary.png   -- combined 2x3 dashboard

Usage
-----
    cd <project_root>
    python src/benchmark.py --video path/to/video.mp4
    python src/benchmark.py --video path/to/video.mp4 --frames 500 --output reports/custom
    python src/benchmark.py --video path/to/video.mp4 --config src/config.yaml
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import cv2
import numpy as np

# -- Ensure src/ is on the path when run from the project root ----------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config   import load_config
from detector import load_models, resolve_class_ids, run_inference
from selector import AdaptiveModelSelector
from telemetry import (
    get_cpu_ram, get_gpu_stats, frame_complexity,
    compute_motion, extract_detection_stats,
)
from utils import build_feature_vector


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class FrameResult:
    """One frame's benchmark measurements for a single pipeline."""
    pipeline:       str
    frame_idx:      int
    fps:            float
    inference_ms:   float
    cpu_pct:        float
    ram_pct:        float
    gpu_pct:        float
    num_detections: int
    model_tier:     int   # 0=nano, 1=small, 2=medium; fixed pipelines always same


@dataclass
class PipelineSummary:
    """Aggregate statistics computed after a pipeline run."""
    pipeline:        str
    mean_fps:        float
    p95_latency_ms:  float
    mean_latency_ms: float
    mean_cpu:        float
    mean_ram:        float
    mean_gpu:        float
    mean_detections: float
    frames_run:      int
    tier_pct:        Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Pipeline runners
# -----------------------------------------------------------------------------

def _run_fixed_pipeline(
    pipeline_name: str,
    model,
    tier_idx: int,
    cap: cv2.VideoCapture,
    max_frames: int,
) -> List[FrameResult]:
    """
    Run a fixed-model pipeline (always uses the same YOLO variant).

    Args:
        pipeline_name: Human-readable label (e.g. "Fixed-N").
        model:         Loaded YOLO model to use for every frame.
        tier_idx:      Tier index (0=nano, 1=small, 2=medium).
        cap:           Rewound VideoCapture object.
        max_frames:    Maximum number of frames to process.

    Returns:
        List of FrameResult records, one per processed frame.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results: List[FrameResult] = []
    frame_idx = 0

    print(f"[Benchmark] Running '{pipeline_name}' ...")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        yolo_results = run_inference(model, frame)
        t1 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        fps    = 1000.0 / inf_ms if inf_ms > 0 else 0.0

        num_det, _, _, _ = extract_detection_stats(yolo_results)
        cpu_pct, ram_pct = get_cpu_ram()
        gpu_pct, _       = get_gpu_stats()

        results.append(FrameResult(
            pipeline       = pipeline_name,
            frame_idx      = frame_idx,
            fps            = fps,
            inference_ms   = inf_ms,
            cpu_pct        = cpu_pct,
            ram_pct        = ram_pct,
            gpu_pct        = gpu_pct,
            num_detections = num_det,
            model_tier     = tier_idx,
        ))
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  {pipeline_name}: {frame_idx}/{max_frames} frames  "
                  f"FPS={fps:.1f}  lat={inf_ms:.1f}ms")

    print(f"  {pipeline_name}: done ({frame_idx} frames)")
    return results


def _run_adaptive_pipeline(
    cfg,
    models: dict,
    cap: cv2.VideoCapture,
    max_frames: int,
) -> List[FrameResult]:
    """
    Run the adaptive pipeline using the MLP-backed model selector.

    The selector starts fresh (no pre-loaded weights) so the benchmark
    reflects cold-start learning behaviour rather than a pre-trained state.

    Args:
        cfg:        Application config (selector hyper-parameters, hysteresis).
        models:     Dict of loaded YOLO models keyed by tier label.
        cap:        Rewound VideoCapture object.
        max_frames: Maximum number of frames to process.

    Returns:
        List of FrameResult records, one per processed frame.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model_levels  = cfg.model.tiers
    current_level = 0
    current_model = models[model_levels[current_level]]

    # Fresh selector -- no pre-loaded weights for a fair benchmark
    selector_cfg = cfg.selector
    selector_cfg.weights_path = ""          # disable weight loading
    selector = AdaptiveModelSelector(selector_cfg, n_tiers=len(model_levels))

    hyst              = cfg.hysteresis
    last_switch_time  = time.time() - hyst.min_hold_time
    last_switch_frame = -hyst.min_hold_frames

    fps_window = collections.deque(maxlen=10)
    det_ema    = 0.0
    DET_EMA_A  = 0.15
    prev_gray  = None

    results: List[FrameResult] = []
    frame_idx = 0

    print("[Benchmark] Running 'Adaptive' ...")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        yolo_results = run_inference(current_model, frame)
        t1 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        fps    = 1000.0 / inf_ms if inf_ms > 0 else 0.0
        fps_window.append(fps)
        fps_smooth = float(np.mean(fps_window))

        num_det, avg_conf, max_conf, track_count = extract_detection_stats(yolo_results)
        det_ema = (1 - DET_EMA_A) * det_ema + DET_EMA_A * num_det

        motion, prev_gray = compute_motion(prev_gray, frame)
        cpu_pct, ram_pct  = get_cpu_ram()
        gpu_pct, gpu_mem  = get_gpu_stats()
        complexity        = frame_complexity(frame)

        raw_feat = build_feature_vector(
            cpu_pct, ram_pct, gpu_pct, gpu_mem,
            fps_smooth, inf_ms, complexity,
            current_level,
            num_det, avg_conf, max_conf,
            track_count, det_ema, motion,
        )

        recommended = selector.predict(raw_feat)
        probs       = selector.confidence(raw_feat)
        selector.observe(raw_feat, current_level, fps_smooth, avg_conf, num_det)

        now         = time.time()
        frames_held = frame_idx - last_switch_frame
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
            current_level = recommended
            current_model = models[model_levels[current_level]]
            last_switch_time  = now
            last_switch_frame = frame_idx

        results.append(FrameResult(
            pipeline       = "Adaptive",
            frame_idx      = frame_idx,
            fps            = fps,
            inference_ms   = inf_ms,
            cpu_pct        = cpu_pct,
            ram_pct        = ram_pct,
            gpu_pct        = gpu_pct,
            num_detections = num_det,
            model_tier     = current_level,
        ))
        frame_idx += 1

        if frame_idx % 100 == 0:
            tier_name = model_levels[current_level]
            print(f"  Adaptive: {frame_idx}/{max_frames} frames  "
                  f"tier={tier_name}  FPS={fps:.1f}  lat={inf_ms:.1f}ms")

    print(f"  Adaptive: done ({frame_idx} frames)")
    return results


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def _summarise(records: List[FrameResult]) -> PipelineSummary:
    """Compute aggregate statistics from a list of FrameResult records."""
    fps  = np.array([r.fps            for r in records])
    lat  = np.array([r.inference_ms   for r in records])
    cpu  = np.array([r.cpu_pct        for r in records])
    ram  = np.array([r.ram_pct        for r in records])
    gpu  = np.array([r.gpu_pct        for r in records])
    det  = np.array([r.num_detections for r in records])
    tier = np.array([r.model_tier     for r in records])

    tier_labels = {0: "nano", 1: "small", 2: "medium"}
    n = max(len(records), 1)
    tier_pct = {
        tier_labels[i]: float(100.0 * (tier == i).sum() / n)
        for i in range(3)
    }

    return PipelineSummary(
        pipeline        = records[0].pipeline,
        mean_fps        = float(fps.mean()),
        p95_latency_ms  = float(np.percentile(lat, 95)),
        mean_latency_ms = float(lat.mean()),
        mean_cpu        = float(cpu.mean()),
        mean_ram        = float(ram.mean()),
        mean_gpu        = float(gpu.mean()),
        mean_detections = float(det.mean()),
        frames_run      = len(records),
        tier_pct        = tier_pct,
    )


# -----------------------------------------------------------------------------
# CSV export
# -----------------------------------------------------------------------------

def _save_csv(all_records: List[FrameResult], output_dir: str) -> str:
    """Write all frame records to a CSV file and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = [
        "pipeline", "frame_idx", "fps", "inference_ms",
        "cpu_pct", "ram_pct", "gpu_pct", "num_detections", "model_tier",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_records:
            writer.writerow(asdict(rec))
    print(f"[Benchmark] CSV saved -> '{path}'")
    return path


# -----------------------------------------------------------------------------
# Plot generation
# -----------------------------------------------------------------------------

PIPELINE_COLORS = {
    "Adaptive": "#2196f3",
    "Fixed-N":  "#4caf50",
    "Fixed-S":  "#ff9800",
    "Fixed-M":  "#f44336",
}

PIPELINE_ORDER = ["Adaptive", "Fixed-N", "Fixed-S", "Fixed-M"]


def _group_by_pipeline(
    all_records: List[FrameResult],
) -> Dict[str, List[FrameResult]]:
    groups: Dict[str, List[FrameResult]] = {}
    for rec in all_records:
        groups.setdefault(rec.pipeline, []).append(rec)
    return groups


def generate_plots(
    all_records: List[FrameResult],
    summaries: List[PipelineSummary],
    output_dir: str,
) -> None:
    """
    Generate and save all benchmark comparison plots.

    Plots produced
    --------------
    1. benchmark_fps.png          -- FPS over frames (line) + bar chart of means
    2. benchmark_latency.png      -- latency distribution (overlapping histograms)
    3. benchmark_resources.png    -- mean CPU / RAM / GPU grouped bar chart
    4. benchmark_detections.png   -- mean detections per pipeline bar chart
    5. benchmark_summary.png      -- 2x3 combined dashboard
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to import Axes3D")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
    except ImportError:
        print("[Benchmark] matplotlib not installed -- skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    groups = _group_by_pipeline(all_records)

    def _save(fig, name: str) -> None:
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[Benchmark] Plot saved -> '{path}'")

    # ── 1. FPS over frames + mean bar ─────────────────────────────────────────
    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    for name in PIPELINE_ORDER:
        if name not in groups:
            continue
        recs  = groups[name]
        fps_v = np.array([r.fps for r in recs])
        # Smooth with a 15-frame rolling mean for readability
        kernel = np.ones(15) / 15
        fps_sm = np.convolve(fps_v, kernel, mode="same")
        ax_line.plot(
            range(len(fps_sm)), fps_sm,
            label=name, color=PIPELINE_COLORS[name], linewidth=1.0,
        )

    ax_line.set_xlabel("Frame index")
    ax_line.set_ylabel("FPS (15-frame rolling mean)")
    ax_line.set_title("FPS Over Time by Pipeline")
    ax_line.legend()
    ax_line.grid(alpha=0.3)

    bar_names  = [s.pipeline for s in summaries]
    bar_fps    = [s.mean_fps for s in summaries]
    bar_colors = [PIPELINE_COLORS.get(n, "#888") for n in bar_names]
    bars = ax_bar.bar(bar_names, bar_fps, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, bar_fps):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax_bar.set_ylabel("Mean FPS")
    ax_bar.set_title("Mean FPS Comparison")
    ax_bar.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "benchmark_fps.png")

    # ── 2. Latency distribution (overlapping histograms) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in PIPELINE_ORDER:
        if name not in groups:
            continue
        lat_v = np.array([r.inference_ms for r in groups[name]])
        ax.hist(
            lat_v, bins=60, alpha=0.55,
            label=f"{name} (mean={lat_v.mean():.1f}ms)",
            color=PIPELINE_COLORS[name], edgecolor="none",
        )
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Frame count")
    ax.set_title("Inference Latency Distribution by Pipeline")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "benchmark_latency.png")

    # ── 3. System resources grouped bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(summaries))
    width  = 0.25
    labels = [s.pipeline for s in summaries]

    bars_cpu = ax.bar(x - width, [s.mean_cpu for s in summaries],
                      width, label="CPU %", color="#f44336", alpha=0.85)
    bars_ram = ax.bar(x,         [s.mean_ram for s in summaries],
                      width, label="RAM %", color="#2196f3", alpha=0.85)
    bars_gpu = ax.bar(x + width, [s.mean_gpu for s in summaries],
                      width, label="GPU %", color="#4caf50", alpha=0.85)

    for group in (bars_cpu, bars_ram, bars_gpu):
        for bar in group:
            h = bar.get_height()
            if h > 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean utilisation (%)")
    ax.set_title("Mean System Resource Utilisation by Pipeline")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "benchmark_resources.png")

    # ── 4. Mean detections bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    det_vals   = [s.mean_detections for s in summaries]
    det_labels = [s.pipeline        for s in summaries]
    det_colors = [PIPELINE_COLORS.get(n, "#888") for n in det_labels]
    bars = ax.bar(det_labels, det_vals, color=det_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, det_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Mean detections per frame")
    ax.set_title("Mean Detection Count by Pipeline")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "benchmark_detections.png")

    # ── 5. Summary dashboard (2x3) ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Benchmark Summary -- Adaptive vs Fixed Pipelines", fontsize=14, y=1.01)

    # [0,0] FPS over frames
    for name in PIPELINE_ORDER:
        if name not in groups:
            continue
        fps_v  = np.array([r.fps for r in groups[name]])
        fps_sm = np.convolve(fps_v, np.ones(15) / 15, mode="same")
        axes[0, 0].plot(fps_sm, label=name, color=PIPELINE_COLORS[name], lw=0.9)
    axes[0, 0].set_title("FPS Over Frames (smoothed)")
    axes[0, 0].set_xlabel("Frame index")
    axes[0, 0].set_ylabel("FPS")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # [0,1] Mean FPS bar
    axes[0, 1].bar(
        [s.pipeline for s in summaries],
        [s.mean_fps for s in summaries],
        color=[PIPELINE_COLORS.get(s.pipeline, "#888") for s in summaries],
        edgecolor="white", linewidth=0.5,
    )
    axes[0, 1].set_title("Mean FPS")
    axes[0, 1].set_ylabel("FPS")
    axes[0, 1].grid(alpha=0.3, axis="y")

    # [0,2] Latency histograms
    for name in PIPELINE_ORDER:
        if name not in groups:
            continue
        lat_v = np.array([r.inference_ms for r in groups[name]])
        axes[0, 2].hist(lat_v, bins=50, alpha=0.5,
                        label=name, color=PIPELINE_COLORS[name], edgecolor="none")
    axes[0, 2].set_title("Latency Distribution")
    axes[0, 2].set_xlabel("ms")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.3, axis="y")

    # [1,0] Resources grouped bar
    x     = np.arange(len(summaries))
    w     = 0.25
    axes[1, 0].bar(x - w, [s.mean_cpu for s in summaries], w, label="CPU", color="#f44336", alpha=0.85)
    axes[1, 0].bar(x,     [s.mean_ram for s in summaries], w, label="RAM", color="#2196f3", alpha=0.85)
    axes[1, 0].bar(x + w, [s.mean_gpu for s in summaries], w, label="GPU", color="#4caf50", alpha=0.85)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([s.pipeline for s in summaries], fontsize=8)
    axes[1, 0].set_title("Mean Resource Utilisation (%)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3, axis="y")

    # [1,1] Detections bar
    axes[1, 1].bar(
        [s.pipeline for s in summaries],
        [s.mean_detections for s in summaries],
        color=[PIPELINE_COLORS.get(s.pipeline, "#888") for s in summaries],
        edgecolor="white", linewidth=0.5,
    )
    axes[1, 1].set_title("Mean Detections per Frame")
    axes[1, 1].set_ylabel("Detections")
    axes[1, 1].grid(alpha=0.3, axis="y")

    # [1,2] Summary stats table
    axes[1, 2].axis("off")
    col_labels = ["Pipeline", "Mean FPS", "P95 Lat (ms)", "Mean CPU%", "Mean Det"]
    table_data = [
        [
            s.pipeline,
            f"{s.mean_fps:.1f}",
            f"{s.p95_latency_ms:.1f}",
            f"{s.mean_cpu:.1f}",
            f"{s.mean_detections:.2f}",
        ]
        for s in summaries
    ]
    tbl = axes[1, 2].table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    axes[1, 2].set_title("Summary Statistics")

    fig.tight_layout()
    _save(fig, "benchmark_summary.png")

    print(f"[Benchmark] All plots saved to '{output_dir}/'")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Adaptive vs Fixed YOLO pipelines on a local video file.\n\n"
            "Compares: Adaptive (MLP selector), Fixed-N (nano), "
            "Fixed-S (small), Fixed-M (medium)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        metavar="PATH",
        help="Path to the input video file (required).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        metavar="N",
        help="Number of frames to process per pipeline (default: 300).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a custom config.yaml (default: src/config.yaml).",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default="reports",
        help="Directory to save CSV and plot files (default: reports).",
    )
    args = parser.parse_args()

    # -- Validate video file ---------------------------------------------------
    if not os.path.isfile(args.video):
        print(f"[Benchmark] Error: video file not found: '{args.video}'")
        sys.exit(1)

    # -- Load config and models ------------------------------------------------
    cfg    = load_config(args.config)
    models = load_models(cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Benchmark] Error: could not open video '{args.video}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native   = cap.get(cv2.CAP_PROP_FPS)
    w_in         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames   = min(args.frames, total_frames) if total_frames > 0 else args.frames

    print(
        f"[Benchmark] Video: '{args.video}'  {w_in}x{h_in}  "
        f"{fps_native:.1f} fps  {total_frames} frames\n"
        f"[Benchmark] Processing {max_frames} frames per pipeline\n"
        f"[Benchmark] Output directory: '{args.output}'\n"
    )

    # -- Run all pipelines -----------------------------------------------------
    all_records: List[FrameResult] = []

    # Adaptive
    adaptive_records = _run_adaptive_pipeline(cfg, models, cap, max_frames)
    all_records.extend(adaptive_records)

    # Fixed pipelines
    tier_map = [
        ("Fixed-N", "n", 0),
        ("Fixed-S", "s", 1),
        ("Fixed-M", "m", 2),
    ]
    for name, tier_key, tier_idx in tier_map:
        if tier_key not in models:
            print(f"[Benchmark] Warning: model tier '{tier_key}' not loaded -- skipping {name}.")
            continue
        records = _run_fixed_pipeline(name, models[tier_key], tier_idx, cap, max_frames)
        all_records.extend(records)

    cap.release()

    # -- Compute summaries -----------------------------------------------------
    groups    = _group_by_pipeline(all_records)
    summaries = [_summarise(groups[p]) for p in PIPELINE_ORDER if p in groups]

    # -- Print summary table ---------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Pipeline':<12} {'Mean FPS':>9} {'P95 Lat':>9} {'Mean Lat':>9} "
          f"{'CPU%':>7} {'RAM%':>7} {'GPU%':>7} {'Det':>6}")
    print("-" * 70)
    for s in summaries:
        print(
            f"{s.pipeline:<12} {s.mean_fps:>9.1f} {s.p95_latency_ms:>9.1f} "
            f"{s.mean_latency_ms:>9.1f} {s.mean_cpu:>7.1f} {s.mean_ram:>7.1f} "
            f"{s.mean_gpu:>7.1f} {s.mean_detections:>6.2f}"
        )
        if s.tier_pct:
            tier_str = "  ".join(f"{k}:{v:.0f}%" for k, v in s.tier_pct.items() if v > 0)
            print(f"{'':12}   Tier usage: {tier_str}")
    print("=" * 70 + "\n")

    # -- Save CSV and plots ----------------------------------------------------
    _save_csv(all_records, args.output)
    generate_plots(all_records, summaries, args.output)

    print("[Benchmark] Complete.")


if __name__ == "__main__":
    main()
