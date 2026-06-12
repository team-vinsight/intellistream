# Migration Summary — Adaptive YOLO Refactor

## Overview

`yolo-claude-2.py` (single 500-line monolith) has been refactored into a
modular, maintainable package under `src/`.  All original behaviour is
preserved; the changes listed below are additive or drop-in replacements.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point — orchestrates the main capture-infer-select-render loop |
| `src/config.py` | Typed configuration loader (reads `config.yaml` via PyYAML) |
| `src/config.yaml` | Human-editable configuration file — **no source code changes needed** |
| `src/selector.py` | `AdaptiveModelSelector` + `TierPerformanceTracker` (PyTorch MLP backend) |
| `src/detector.py` | YOLO model loading, class-ID resolution, inference, masked-frame renderer |
| `src/telemetry.py` | CPU/RAM/GPU stats, frame complexity, motion score, detection statistics |
| `src/metrics.py` | Per-frame metrics collection, CSV persistence, Matplotlib graph generation |
| `src/visualization.py` | HUD overlay renderer (extracted from the original monolith) |
| `src/camera.py` | Camera probing, opening, and frame-validity guard |
| `src/utils.py` | Shared helpers: `build_feature_vector`, `placeholder_frame` |
| `src/reports/` | Output directory for CSV and PNG graph files |

## Files Modified

| File | Change |
|------|--------|
| `src/yolo-claude-2.py` | **Unchanged** — kept as the original reference implementation |

---

## Architectural Changes

### 1 — PyTorch MLP (replaces NumPy hand-rolled backprop)

**Before (`yolo-claude-2.py`):**
- Manual weight matrices (`W1`, `b1`, `W2`, `b2`) as NumPy arrays
- Hand-coded forward pass, backward pass, and SGD+momentum update
- Weights persisted via `pickle` (`.pkl` file)

**After (`selector.py`):**
- `SelectorMLP(nn.Module)` — standard `nn.Sequential(Linear, ReLU, Linear)`
- `torch.optim.SGD(momentum=...)` handles the update step
- `nn.CrossEntropyLoss` replaces the manual softmax + cross-entropy gradient
- Weights persisted via `torch.save` / `torch.load` (`.pt` checkpoint)
- He initialisation (`nn.init.kaiming_normal_`) matches the original `sqrt(2/fan_in)` scheme
- All public methods (`predict`, `confidence`, `observe`, `save_weights`) have identical signatures

### 2 — Configurable class filtering (`config.yaml` + `detector.py`)

- `config.yaml` → `detection.classes` accepts a list of COCO class **names**
- `resolve_class_ids()` in `detector.py` converts names to integer IDs at startup
  using the model's own `model.names` dictionary
- `extract_detection_stats()` in `telemetry.py` accepts an optional `class_ids` list
- `render_masked_frame()` in `detector.py` applies the same filter before masking
- **Changing the class list requires only editing `config.yaml`** — no source changes

### 3 — Bounding-box masking (replaces `results[0].plot()`)

**Before:** `results[0].plot()` drew YOLO bounding boxes, class labels, and
confidence scores on the frame.

**After:** `render_masked_frame()` in `detector.py`:
- Copies the original frame (no YOLO annotations)
- Iterates over filtered detections
- Draws a solid black `cv2.rectangle(..., thickness=-1)` over each detected region
- Resizes to `target_width` preserving aspect ratio

### 4 — Performance metrics (`metrics.py`)

- `MetricsCollector.record()` is called once per frame inside the main loop
- On exit, `MetricsCollector.save()` writes a CSV to `src/reports/metrics.csv`
- `MetricsCollector.generate_reports()` produces six PNG graphs:

| Graph | File |
|-------|------|
| FPS over time | `fps_over_time.png` |
| Inference latency histogram | `inference_latency.png` |
| CPU / RAM / GPU over time | `system_resources.png` |
| Detection count over time | `detections_over_time.png` |
| Active model tier over time | `model_tier_over_time.png` |
| Combined 2×3 dashboard | `summary_dashboard.png` |

### 5 — Modularisation

| Concern | Module |
|---------|--------|
| Configuration | `config.py` + `config.yaml` |
| YOLO models + masking | `detector.py` |
| MLP selector + performance tracker | `selector.py` |
| System telemetry | `telemetry.py` |
| Metrics + graphs | `metrics.py` |
| HUD overlay | `visualization.py` |
| Camera I/O | `camera.py` |
| Shared utilities | `utils.py` |
| Main loop | `main.py` |

---

## Dependencies Added

| Package | Reason | Install |
|---------|--------|---------|
| `torch` | PyTorch MLP backend | `pip install torch` |
| `PyYAML` | Parse `config.yaml` | `pip install pyyaml` |
| `matplotlib` | Post-session graphs | `pip install matplotlib` |

Existing dependencies (`opencv-python`, `ultralytics`, `psutil`, `numpy`,
`GPUtil`) are unchanged.

Install all at once:

```bash
pip install torch pyyaml matplotlib
```

---

## How to Run

```bash
# From the project root
python src/main.py

# With a custom config file
python src/main.py --config path/to/my_config.yaml
```

### Keyboard shortcuts (unchanged)

| Key | Action |
|-----|--------|
| `q` | Quit and save |
| `s` | Save selector weights immediately |

### Changing detected classes

Edit `src/config.yaml`:

```yaml
detection:
  classes: [person, car, truck]   # any COCO class names
  confidence_threshold: 0.25
```

Restart the application — no code changes required.

### Reports

After each session, graphs and the metrics CSV are written to `src/reports/`:

```
src/reports/
├── metrics.csv
├── fps_over_time.png
├── inference_latency.png
├── system_resources.png
├── detections_over_time.png
├── model_tier_over_time.png
└── summary_dashboard.png
```

### Selector weights

The PyTorch checkpoint is saved to `src/adaptive_selector_weights.pt`
(configurable via `selector.weights_path` in `config.yaml`).  The old
`.pkl` files from the NumPy implementation are no longer used and can be
deleted.
