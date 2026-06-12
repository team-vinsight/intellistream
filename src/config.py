"""
config.py
─────────
Loads and exposes the application configuration from config.yaml.

All tuneable parameters live in config.yaml; this module provides a
typed, validated view of that file so the rest of the codebase never
has to parse YAML directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


# ── Default config file location (same directory as this file) ────────────────
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses — one per logical section of config.yaml
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Paths and tier ordering for the three YOLO model variants."""
    nano:   str = "models/yolo/yolo26n.pt"
    small:  str = "models/yolo/yolo26s.pt"
    medium: str = "models/yolo/yolo26m.pt"
    # Ordered list of tier keys — index == tier level
    tiers:  List[str] = field(default_factory=lambda: ["n", "s", "m"])


@dataclass
class DetectionConfig:
    """
    Object-class filtering.

    ``classes`` is a list of COCO class *names* (strings).  An empty list
    means "detect everything".  The detector module resolves names to
    integer IDs at runtime using the model's own class map.

    Example (config.yaml):
        detection:
          classes: [person, car, truck, bus, motorcycle]
          confidence_threshold: 0.25
    """
    classes:               List[str] = field(default_factory=list)
    confidence_threshold:  float     = 0.25


@dataclass
class SelectorConfig:
    """Hyper-parameters for AdaptiveModelSelector and its MLP."""
    input_dim:          int   = 14
    hidden_dim:         int   = 32
    output_dim:         int   = 3
    learning_rate:      float = 0.01
    momentum:           float = 0.9
    min_samples:        int   = 30
    replay_size:        int   = 400
    batch_size:         int   = 24
    update_interval:    int   = 5
    max_perf_weight:    float = 0.70
    perf_ramp_samples:  int   = 300
    upgrade_fps:        float = 22.0
    downgrade_fps:      float = 10.0
    cpu_heavy:          float = 75.0
    ram_heavy:          float = 85.0
    weights_path:       str   = "src/adaptive_selector_weights.pt"


@dataclass
class HysteresisConfig:
    """Guards against rapid model-tier switching."""
    min_hold_frames: int   = 60
    min_hold_time:   float = 2.0
    min_switch_conf: float = 0.55


@dataclass
class CameraConfig:
    """Camera capture settings."""
    width:          int = 1280
    height:         int = 720
    max_bad_frames: int = 30


@dataclass
class MetricsConfig:
    """Runtime metrics collection and reporting."""
    enabled:      bool = True
    save_path:    str  = "src/reports/metrics.csv"
    reports_dir:  str  = "src/reports"


@dataclass
class AppConfig:
    """Root configuration object — aggregates all sections."""
    model:      ModelConfig      = field(default_factory=ModelConfig)
    detection:  DetectionConfig  = field(default_factory=DetectionConfig)
    selector:   SelectorConfig   = field(default_factory=SelectorConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    camera:     CameraConfig     = field(default_factory=CameraConfig)
    metrics:    MetricsConfig    = field(default_factory=MetricsConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _merge(dataclass_instance, mapping: dict):
    """
    Recursively overwrite dataclass fields from a nested dict.
    Unknown keys are silently ignored so the YAML can be extended
    without breaking older code.
    """
    for key, value in mapping.items():
        if not hasattr(dataclass_instance, key):
            continue
        current = getattr(dataclass_instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dataclass_instance, key, value)


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from *path* (defaults to ``src/config.yaml``).

    Falls back to all-default values if the file does not exist, so the
    application can run without a config file during development.
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    cfg = AppConfig()

    if not os.path.exists(config_path):
        print(f"[Config] '{config_path}' not found — using defaults.")
        return cfg

    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    _merge(cfg, raw)
    print(f"[Config] Loaded from '{config_path}'")
    return cfg
