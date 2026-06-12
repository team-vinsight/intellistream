"""
metrics.py
──────────
Runtime performance metrics collection, CSV persistence, and
post-execution Matplotlib graph generation.

Collected per-frame metrics
────────────────────────────
  timestamp_s      — wall-clock seconds since session start
  fps              — 10-frame rolling mean FPS
  inference_ms     — raw inference latency in milliseconds
  cpu_pct          — CPU utilisation (%)
  ram_pct          — RAM utilisation (%)
  gpu_pct          — GPU load (%)
  num_detections   — filtered detection count
  model_tier       — current tier index (0 = nano, 1 = small, 2 = medium)

After the session ends, ``generate_reports()`` writes PNG graphs into the
configured ``reports_dir``.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np

from config import MetricsConfig


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame metric record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameMetric:
    """One row of collected data — one entry per processed frame."""
    timestamp_s:    float
    fps:            float
    inference_ms:   float
    cpu_pct:        float
    ram_pct:        float
    gpu_pct:        float
    num_detections: int
    model_tier:     int


# ─────────────────────────────────────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Accumulates ``FrameMetric`` records in memory and flushes them to a CSV
    file at the end of the session.

    Usage::

        collector = MetricsCollector(cfg.metrics)
        # inside the main loop:
        collector.record(fps=..., inference_ms=..., ...)
        # after the loop:
        collector.save()
        collector.generate_reports()
    """

    _FIELDNAMES = [
        "timestamp_s", "fps", "inference_ms",
        "cpu_pct", "ram_pct", "gpu_pct",
        "num_detections", "model_tier",
    ]

    def __init__(self, cfg: MetricsConfig) -> None:
        self.cfg        = cfg
        self._records:  List[FrameMetric] = []
        self._start_t   = time.time()

    def record(
        self,
        fps:            float,
        inference_ms:   float,
        cpu_pct:        float,
        ram_pct:        float,
        gpu_pct:        float,
        num_detections: int,
        model_tier:     int,
    ) -> None:
        """Append one frame's metrics to the in-memory buffer."""
        if not self.cfg.enabled:
            return
        self._records.append(FrameMetric(
            timestamp_s    = time.time() - self._start_t,
            fps            = fps,
            inference_ms   = inference_ms,
            cpu_pct        = cpu_pct,
            ram_pct        = ram_pct,
            gpu_pct        = gpu_pct,
            num_detections = num_detections,
            model_tier     = model_tier,
        ))

    def save(self) -> None:
        """Write all collected records to the configured CSV file."""
        if not self.cfg.enabled or not self._records:
            return
        os.makedirs(os.path.dirname(self.cfg.save_path) or ".", exist_ok=True)
        with open(self.cfg.save_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            for rec in self._records:
                writer.writerow(asdict(rec))
        print(f"[Metrics] Saved {len(self._records)} rows → '{self.cfg.save_path}'")

    # ── Graph generation ──────────────────────────────────────────────────────

    def generate_reports(self) -> None:
        """
        Produce and save Matplotlib PNG graphs into ``cfg.reports_dir``.

        Graphs generated
        ─────────────────
        1. fps_over_time.png          — rolling FPS vs time
        2. inference_latency.png      — inference latency histogram
        3. system_resources.png       — CPU / RAM / GPU over time
        4. detections_over_time.png   — detection count vs time
        5. model_tier_over_time.png   — active tier vs time
        6. summary_dashboard.png      — 2 × 3 combined dashboard
        """
        if not self.cfg.enabled or not self._records:
            print("[Metrics] No data to plot.")
            return

        # Lazy import — Matplotlib is only needed at report time.
        # Suppress the spurious Axes3D warning that fires when both a
        # system-level and a venv-level Matplotlib are present; the 3D
        # projection is not used anywhere in this project.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Unable to import Axes3D",
                    category=UserWarning,
                )
                import matplotlib
                matplotlib.use("Agg")       # headless backend — no display needed
                import matplotlib.pyplot as plt
                import matplotlib.ticker as ticker
        except ImportError:
            print("[Metrics] matplotlib not installed — skipping graphs.")
            return

        os.makedirs(self.cfg.reports_dir, exist_ok=True)

        # ── Unpack arrays ─────────────────────────────────────────────
        t   = np.array([r.timestamp_s    for r in self._records])
        fps = np.array([r.fps            for r in self._records])
        lat = np.array([r.inference_ms   for r in self._records])
        cpu = np.array([r.cpu_pct        for r in self._records])
        ram = np.array([r.ram_pct        for r in self._records])
        gpu = np.array([r.gpu_pct        for r in self._records])
        det = np.array([r.num_detections for r in self._records])
        tier= np.array([r.model_tier     for r in self._records])

        TIER_LABELS = {0: "nano", 1: "small", 2: "medium"}
        TIER_COLORS = {0: "#4caf50", 1: "#ff9800", 2: "#f44336"}

        # ── Helper: save figure ───────────────────────────────────────
        def _save(fig: "plt.Figure", name: str) -> None:
            path = os.path.join(self.cfg.reports_dir, name)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[Metrics] Graph → '{path}'")

        # ── 1. FPS over time ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, fps, color="#2196f3", linewidth=0.8, label="FPS")
        ax.axhline(fps.mean(), color="#f44336", linestyle="--",
                   linewidth=1, label=f"Mean {fps.mean():.1f}")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("FPS")
        ax.set_title("Frames Per Second Over Time")
        ax.legend(); ax.grid(alpha=0.3)
        _save(fig, "fps_over_time.png")

        # ── 2. Inference latency histogram ────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lat, bins=50, color="#9c27b0", edgecolor="white", linewidth=0.4)
        ax.axvline(lat.mean(), color="#f44336", linestyle="--",
                   label=f"Mean {lat.mean():.1f} ms")
        ax.axvline(np.percentile(lat, 95), color="#ff9800", linestyle="--",
                   label=f"P95 {np.percentile(lat, 95):.1f} ms")
        ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Frame count")
        ax.set_title("Inference Latency Distribution")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        _save(fig, "inference_latency.png")

        # ── 3. System resources over time ─────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, cpu, label="CPU %",  color="#f44336", linewidth=0.8)
        ax.plot(t, ram, label="RAM %",  color="#2196f3", linewidth=0.8)
        ax.plot(t, gpu, label="GPU %",  color="#4caf50", linewidth=0.8)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Utilisation (%)")
        ax.set_title("System Resource Utilisation Over Time")
        ax.legend(); ax.grid(alpha=0.3)
        _save(fig, "system_resources.png")

        # ── 4. Detections over time ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, det, color="#ff9800", linewidth=0.8)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Detection count")
        ax.set_title("Filtered Detection Count Over Time")
        ax.grid(alpha=0.3)
        _save(fig, "detections_over_time.png")

        # ── 5. Model tier over time ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 3))
        for lvl, color in TIER_COLORS.items():
            mask = tier == lvl
            if mask.any():
                ax.fill_between(t, lvl - 0.4, lvl + 0.4,
                                where=mask, color=color, alpha=0.7,
                                label=TIER_LABELS[lvl])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([TIER_LABELS[i] for i in range(3)])
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Active tier")
        ax.set_title("Active Model Tier Over Time")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3, axis="x")
        _save(fig, "model_tier_over_time.png")

        # ── 6. Summary dashboard (2 × 3 grid) ────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle("Adaptive YOLO — Session Summary", fontsize=14, y=1.01)

        # [0,0] FPS
        axes[0, 0].plot(t, fps, color="#2196f3", linewidth=0.7)
        axes[0, 0].axhline(fps.mean(), color="#f44336", linestyle="--", linewidth=1)
        axes[0, 0].set_title(f"FPS  (mean={fps.mean():.1f})")
        axes[0, 0].set_xlabel("Time (s)"); axes[0, 0].grid(alpha=0.3)

        # [0,1] Latency histogram
        axes[0, 1].hist(lat, bins=40, color="#9c27b0", edgecolor="white", lw=0.3)
        axes[0, 1].axvline(lat.mean(), color="#f44336", linestyle="--", linewidth=1)
        axes[0, 1].set_title(f"Latency  (mean={lat.mean():.1f} ms)")
        axes[0, 1].set_xlabel("ms"); axes[0, 1].grid(alpha=0.3, axis="y")

        # [0,2] CPU / RAM / GPU
        axes[0, 2].plot(t, cpu, label="CPU", color="#f44336", lw=0.7)
        axes[0, 2].plot(t, ram, label="RAM", color="#2196f3", lw=0.7)
        axes[0, 2].plot(t, gpu, label="GPU", color="#4caf50", lw=0.7)
        axes[0, 2].set_ylim(0, 105)
        axes[0, 2].set_title("System Resources (%)")
        axes[0, 2].set_xlabel("Time (s)")
        axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

        # [1,0] Detections
        axes[1, 0].plot(t, det, color="#ff9800", lw=0.7)
        axes[1, 0].set_title(f"Detections  (mean={det.mean():.1f})")
        axes[1, 0].set_xlabel("Time (s)"); axes[1, 0].grid(alpha=0.3)

        # [1,1] Tier timeline
        for lvl, color in TIER_COLORS.items():
            mask = tier == lvl
            if mask.any():
                axes[1, 1].fill_between(t, lvl - 0.4, lvl + 0.4,
                                        where=mask, color=color, alpha=0.7,
                                        label=TIER_LABELS[lvl])
        axes[1, 1].set_yticks([0, 1, 2])
        axes[1, 1].set_yticklabels([TIER_LABELS[i] for i in range(3)])
        axes[1, 1].set_title("Active Model Tier")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3, axis="x")

        # [1,2] Summary stats table
        axes[1, 2].axis("off")
        tier_counts = {TIER_LABELS[i]: int((tier == i).sum()) for i in range(3)}
        total = len(self._records)
        table_data = [
            ["Metric", "Value"],
            ["Total frames",    str(total)],
            ["Session time",    f"{t[-1]:.1f} s"],
            ["Mean FPS",        f"{fps.mean():.1f}"],
            ["P95 latency",     f"{np.percentile(lat, 95):.1f} ms"],
            ["Mean CPU",        f"{cpu.mean():.1f} %"],
            ["Mean RAM",        f"{ram.mean():.1f} %"],
            ["Mean GPU",        f"{gpu.mean():.1f} %"],
            ["Mean detections", f"{det.mean():.1f}"],
        ] + [
            [f"  Tier {k}", f"{v} frames ({100*v/max(total,1):.1f}%)"]
            for k, v in tier_counts.items()
        ]
        tbl = axes[1, 2].table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.4)
        axes[1, 2].set_title("Session Summary")

        fig.tight_layout()
        _save(fig, "summary_dashboard.png")

        print(f"[Metrics] All graphs saved to '{self.cfg.reports_dir}/'")
