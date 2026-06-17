# IntelliStream

Real-time adaptive object detection and masking system built on YOLO and a PyTorch MLP selector. The system dynamically switches between YOLO nano, small, and medium variants at runtime to balance inference speed against detection quality based on live system telemetry and scene complexity.

---

## Overview

IntelliStream runs a continuous video pipeline that:

1. Reads frames from a webcam or a local video file
2. Runs YOLO inference on the active model tier
3. Feeds 14 system and scene features into an online-trained MLP
4. Switches model tiers when the MLP is confident a change improves the speed/quality trade-off
5. Masks detected objects with solid black rectangles and overlays a live HUD
6. Saves per-frame metrics and generates performance graphs on exit

---

## Repository Structure

    intellistream/
    models/yolo/
        yolo26n.pt
        yolo26s.pt
        yolo26m.pt
    src/
        main.py             Application entry point
        benchmark.py        Pipeline benchmark script
        config.py           Typed config loader
        config.yaml         All tuneable parameters
        camera.py           Webcam initialisation helpers
        detector.py         YOLO loading, inference, masked rendering
        selector.py         AdaptiveModelSelector (PyTorch MLP)
        telemetry.py        CPU / RAM / GPU / motion / complexity
        metrics.py          Per-frame metrics collection and graphs
        utils.py            Feature vector builder, placeholder frame
        visualization.py    HUD overlay renderer
    scripts/                Utility and export scripts
    reports/                Auto-generated CSV and PNG reports
    README.md

---

## Requirements

- Python 3.10+
- ultralytics
- torch
- opencv-python
- psutil
- matplotlib
- gputil (optional, enables GPU telemetry)

## Create Virtual Environment

    python -m venv venv

Windows Command Prompt (CMD)

    .venv\Scripts\activate

Windows PowerShell

    .\.venv\Scripts\Activate.ps1

Linux/macOS

    source venv/bin/activate

Install all dependencies:

    pip install ultralytics torch opencv-python psutil matplotlib gputil

---


## Running the Main Application

All commands are run from the project root.

#### Live webcam (default)

    python src/main.py

#### Specific webcam index

    python src/main.py --source 1

#### Local video file

    python src/main.py --source path/to/video.mp4

#### Custom configuration file

    python src/main.py --config src/config.yaml

#### Video file with a custom config

    python src/main.py --source path/to/video.mp4 --config src/config.yaml

#### Keyboard controls

| Key | Action |
|-----|--------|
| q   | Quit and save reports |
| s   | Save MLP selector weights immediately |

On exit the application saves to the configured reports directory:

| File | Contents |
|------|----------|
| metrics.csv | Per-frame telemetry |
| fps_over_time.png | Rolling FPS vs time |
| inference_latency.png | Latency histogram |
| system_resources.png | CPU / RAM / GPU over time |
| detections_over_time.png | Detection count vs time |
| model_tier_over_time.png | Active tier vs time |
| summary_dashboard.png | Combined 2x3 dashboard |

---

## Running the Benchmark

The benchmark evaluates four pipelines on a local video file and produces comparison plots.
A webcam cannot be used as the benchmark source.

#### Basic usage (300 frames per pipeline)

    python src/benchmark.py --video path/to/video.mp4

#### Custom frame count

    python src/benchmark.py --video path/to/video.mp4 --frames 500

#### Custom output directory

    python src/benchmark.py --video path/to/video.mp4 --output reports/run1

#### Custom config and output directory

    python src/benchmark.py --video path/to/video.mp4 --config src/config.yaml --output reports/run

#### All options

    usage: benchmark.py [-h] --video PATH [--frames N] [--config PATH] [--output DIR]

      --video   PATH   Path to the input video file (required)
      --frames  N      Frames to process per pipeline (default: 300)
      --config  PATH   Path to a custom config.yaml (default: src/config.yaml)
      --output  DIR    Directory for CSV and plot output (default: reports)

#### Pipelines compared

| Pipeline | Description |
|----------|-------------|
| Adaptive | MLP selector switches between nano / small / medium dynamically |
| Fixed-N  | Always uses YOLO nano |
| Fixed-S  | Always uses YOLO small |
| Fixed-M  | Always uses YOLO medium |

#### Output files

| File | Contents |
|------|----------|
| benchmark_results.csv | Raw per-frame data for all pipelines |
| benchmark_fps.png | FPS over frames (line) and mean FPS bar chart |
| benchmark_latency.png | Overlapping inference latency histograms |
| benchmark_resources.png | Mean CPU / RAM / GPU grouped bar chart |
| benchmark_detections.png | Mean detection count per pipeline |
| benchmark_summary.png | Combined 2x3 dashboard |

---

## Configuration

All tuneable parameters live in src/config.yaml.

| Section | Purpose |
|---------|---------|
| model | Paths to the three YOLO weight files and tier ordering |
| detection | COCO class filter list and confidence threshold |
| selector | MLP dimensions, learning rate, replay buffer, FPS thresholds |
| hysteresis | Minimum frames and seconds between tier switches |
| camera | Capture resolution and bad-frame tolerance |
| metrics | Enable/disable collection, CSV path, reports directory |

---

## MLP Selector Feature Vector

The 14-dimensional input fed to the selector MLP every frame:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | cpu_pct | CPU utilisation (%) |
| 1 | ram_pct | RAM utilisation (%) |
| 2 | gpu_pct | GPU load (%) |
| 3 | gpu_mem_pct | GPU memory utilisation (%) |
| 4 | fps_smooth | 10-frame rolling mean FPS |
| 5 | inf_ms | Raw inference latency (ms) |
| 6 | complexity | tanh-normalised Laplacian variance |
| 7 | current_level | Active tier index (0 / 1 / 2) |
| 8 | num_det | Filtered detection count |
| 9 | avg_conf | Mean confidence of filtered detections |
| 10 | max_conf | Maximum confidence |
| 11 | track_count | Detections with confidence >= 0.5 |
| 12 | det_ema | EMA of detection count (temporal stability) |
| 13 | motion | Normalised inter-frame motion score |
