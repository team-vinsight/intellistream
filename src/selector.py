"""
selector.py
───────────
Adaptive model-tier selector backed by a PyTorch MLP.

Architecture: 14 → 32 → 3  (ReLU hidden, Softmax output)
Training:     online mini-batch SGD with momentum, experience-replay buffer
Persistence:  PyTorch checkpoint (.pt) — replaces the old pickle approach

The public API is identical to the original NumPy implementation so that
main.py requires no changes beyond the import.
"""

from __future__ import annotations

import collections
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import SelectorConfig


# ─────────────────────────────────────────────────────────────────────────────
# Per-tier performance tracker  ←  the reward signal
# ─────────────────────────────────────────────────────────────────────────────

class TierPerformanceTracker:
    """
    Maintains an exponential moving average (EMA) composite score for every
    model tier so the selector can learn *empirically* when a heavier model
    is worth the FPS cost.

    Composite score formula
    ───────────────────────
    composite(tier) =
        FPS_W  × clip(fps / TARGET_FPS,  0, 1)
      + DET_W  × clip(avg_conf × log1p(n_det) / log1p(10),  0, 1)

    This encodes the trade-off the rule-based system cannot express:
        "YOLO-M is worth it only when the scene has many confident detections."
    """

    EMA_ALPHA         = 0.08   # slow EMA — brief spikes should not dominate
    FPS_WEIGHT        = 0.55
    DET_WEIGHT        = 0.45
    TARGET_FPS        = 25.0
    MIN_OBS_FOR_LABEL = 15     # need this many obs before trusting a tier's EMA
    SIGNIFICANT_DELTA = 0.04   # must beat current tier by this margin to prefer it

    def __init__(self, n_tiers: int) -> None:
        self.n     = n_tiers
        self.ema   = [0.5] * n_tiers   # initialise to neutral 0.5
        self.count = [0]   * n_tiers

    def composite(self, fps: float, avg_conf: float, num_det: int) -> float:
        """Compute a single scalar quality score in [0, 1]."""
        fps_score = float(np.clip(fps / self.TARGET_FPS, 0.0, 1.0))
        det_score = float(np.clip(
            avg_conf * np.log1p(num_det) / np.log1p(10), 0.0, 1.0))
        return self.FPS_WEIGHT * fps_score + self.DET_WEIGHT * det_score

    def update(self, tier: int, fps: float, avg_conf: float, num_det: int) -> None:
        """Record one observation for *tier* and update its EMA score."""
        s = self.composite(fps, avg_conf, num_det)
        if self.count[tier] == 0:
            self.ema[tier] = s
        else:
            self.ema[tier] = (1 - self.EMA_ALPHA) * self.ema[tier] + self.EMA_ALPHA * s
        self.count[tier] += 1

    def best_adjacent_tier(self, current: int) -> Optional[int]:
        """
        Return a neighbour tier (current ± 1) if it has enough observations
        AND significantly outscores the current tier.

        Returns None → stay at current tier.
        Only considers adjacent tiers to prevent large jumps (n → m directly).
        """
        best, best_score = None, self.ema[current]
        for candidate in (current - 1, current + 1):
            if not (0 <= candidate < self.n):
                continue
            if self.count[candidate] < self.MIN_OBS_FOR_LABEL:
                continue
            if self.ema[candidate] > best_score + self.SIGNIFICANT_DELTA:
                best, best_score = candidate, self.ema[candidate]
        return best


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch MLP
# ─────────────────────────────────────────────────────────────────────────────

class SelectorMLP(nn.Module):
    """
    Two-layer MLP: input_dim → hidden_dim → output_dim.

    Activation: ReLU (hidden), Softmax (output — applied externally via
    CrossEntropyLoss during training, or manually during inference).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # He initialisation — matches the original sqrt(2/fan_in) scheme
        nn.init.kaiming_normal_(self.net[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.net[0].bias)
        nn.init.kaiming_normal_(self.net[2].weight, nonlinearity="relu")
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no softmax) — used with CrossEntropyLoss."""
        return self.net(x)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for inference."""
        return torch.softmax(self.forward(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive model selector — online MLP (PyTorch backend)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveModelSelector:
    """
    Online-learning model-tier selector.

    The MLP is trained incrementally via mini-batch SGD with momentum drawn
    from an experience-replay buffer.  Label generation passes through three
    phases:

    1. Warm-up  (< min_samples):
         Pure rule-based labels only.
    2. Transition (min_samples → perf_ramp_samples):
         Linearly growing mix: rule labels + TierPerformanceTracker labels.
    3. Mature   (> perf_ramp_samples):
         Up to max_perf_weight of labels come from observed performance,
         so the network learns *beyond* the hand-crafted rules.

    Input features (14 dimensions)
    ───────────────────────────────
      [0]  cpu_percent
      [1]  ram_percent
      [2]  gpu_percent
      [3]  gpu_mem_percent
      [4]  current_fps          10-frame rolling mean
      [5]  inference_time_ms
      [6]  frame_complexity     tanh-normalised Laplacian variance
      [7]  current_model_idx    0 / 1 / 2
      [8]  num_detections
      [9]  avg_confidence
      [10] max_confidence
      [11] track_count          detections with conf > 0.5  (reliable objects)
      [12] track_age            EMA of num_detections        (temporal stability)
      [13] motion_score         normalised frame-diff mean
    """

    def __init__(self, cfg: SelectorConfig, n_tiers: int = 3) -> None:
        self.cfg     = cfg
        self.n_tiers = n_tiers

        # ── PyTorch MLP + SGD with momentum ──────────────────────────
        self.device = torch.device("cpu")   # inference is fast enough on CPU
        self.mlp    = SelectorMLP(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
        self.mlp.to(self.device)
        self.optimizer = optim.SGD(
            self.mlp.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── Experience replay ─────────────────────────────────────────
        self.replay_buffer: collections.deque = collections.deque(
            maxlen=cfg.replay_size)
        self.sample_count  = 0
        self.update_ticker = 0

        # ── Welford online normalisation ──────────────────────────────
        self.feat_mean = np.zeros(cfg.input_dim, dtype=np.float64)
        self.feat_var  = np.ones(cfg.input_dim,  dtype=np.float64)
        self.feat_n    = 0

        # ── Per-tier performance reward signal ────────────────────────
        self.perf = TierPerformanceTracker(n_tiers)

        # ── Expose constants used by the HUD ─────────────────────────
        self.MIN_SAMPLES = cfg.min_samples

        self._load_weights()

    # ── Welford online normalisation ──────────────────────────────────────────

    def _update_norm(self, x: np.ndarray) -> None:
        """Update running mean and variance using Welford's algorithm."""
        self.feat_n += 1
        d = x - self.feat_mean
        self.feat_mean += d / self.feat_n
        self.feat_var  += d * (x - self.feat_mean)

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Return a zero-mean, unit-variance version of *x*."""
        std = np.sqrt(self.feat_var / max(self.feat_n, 1)) + 1e-8
        return ((x - self.feat_mean) / std).astype(np.float32)

    # ── Label generation ──────────────────────────────────────────────────────

    def _rule_label(self, raw: np.ndarray, level: int) -> int:
        """
        Heuristic label used during warm-up and as a fallback.

        Downgrades when the system is under pressure or FPS is too low;
        upgrades when FPS headroom exists and resources are free.
        """
        fps, cpu, ram = float(raw[4]), float(raw[0]), float(raw[1])
        pressure = cpu > self.cfg.cpu_heavy or ram > self.cfg.ram_heavy
        if pressure or fps < self.cfg.downgrade_fps:
            return max(0, level - 1)
        if fps > self.cfg.upgrade_fps and not pressure:
            return min(self.n_tiers - 1, level + 1)
        return level

    def _blended_label(
        self,
        raw: np.ndarray,
        level: int,
        fps: float,
        avg_conf: float,
        num_det: int,
    ) -> int:
        """
        Three-phase label generation (see class docstring).

        Always records the observation in the performance tracker regardless
        of which phase is active.
        """
        rule = self._rule_label(raw, level)
        self.perf.update(level, fps, avg_conf, num_det)

        if self.sample_count < self.cfg.min_samples:
            return rule                         # phase 1 — warm-up

        perf_best = self.perf.best_adjacent_tier(level)
        if perf_best is None:
            return rule                         # performance says "stay put"

        perf_w = min(
            self.sample_count / self.cfg.perf_ramp_samples,
            self.cfg.max_perf_weight,
        )
        return perf_best if np.random.random() < perf_w else rule

    # ── Observe + train ───────────────────────────────────────────────────────

    def observe(
        self,
        raw: np.ndarray,
        level: int,
        fps: float,
        avg_conf: float,
        num_det: int,
    ) -> None:
        """
        Called every frame with fresh telemetry and detection outcomes.

        Updates the normaliser, generates a training label, stores the
        experience in the replay buffer, and triggers a mini-batch update
        every ``update_interval`` frames once enough samples have been seen.
        """
        self._update_norm(raw)
        label = self._blended_label(raw, level, fps, avg_conf, num_det)
        self.replay_buffer.append((self._normalise(raw).copy(), label))
        self.sample_count  += 1
        self.update_ticker += 1

        if (
            self.sample_count  >= self.cfg.min_samples
            and self.update_ticker >= self.cfg.update_interval
            and len(self.replay_buffer) >= self.cfg.batch_size
        ):
            self._train_batch()
            self.update_ticker = 0

    def _train_batch(self) -> None:
        """Sample a random mini-batch from the replay buffer and do one SGD step."""
        idx = np.random.choice(
            len(self.replay_buffer), self.cfg.batch_size, replace=False)

        feats  = np.stack([self.replay_buffer[i][0] for i in idx])
        labels = [self.replay_buffer[i][1] for i in idx]

        x = torch.tensor(feats,  dtype=torch.float32, device=self.device)
        y = torch.tensor(labels, dtype=torch.long,    device=self.device)

        self.mlp.train()
        self.optimizer.zero_grad()
        logits = self.mlp(x)
        loss   = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()

    # ── Predict / confidence ──────────────────────────────────────────────────

    def predict(self, raw: np.ndarray) -> int:
        """
        Return the recommended tier index.

        During warm-up (< min_samples) the current level is held to avoid
        premature switching before the MLP has learned anything useful.
        """
        if self.sample_count < self.cfg.min_samples:
            return int(round(float(raw[7])))    # hold current level

        self.mlp.eval()
        with torch.no_grad():
            x = torch.tensor(
                self._normalise(raw), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            probs = self.mlp.probabilities(x).squeeze(0).cpu().numpy()
        return int(np.argmax(probs))

    def confidence(self, raw: np.ndarray) -> np.ndarray:
        """
        Return a probability vector over all tiers.

        During warm-up returns a one-hot vector at the current level.
        """
        if self.sample_count < self.cfg.min_samples:
            p = np.zeros(self.cfg.output_dim, dtype=np.float32)
            p[int(round(float(raw[7])))] = 1.0
            return p

        self.mlp.eval()
        with torch.no_grad():
            x = torch.tensor(
                self._normalise(raw), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            probs = self.mlp.probabilities(x).squeeze(0).cpu().numpy()
        return probs

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_weights(self) -> None:
        """
        Persist the MLP weights, optimiser state, and normaliser statistics
        to a PyTorch checkpoint file.
        """
        os.makedirs(os.path.dirname(self.cfg.weights_path) or ".", exist_ok=True)
        checkpoint = {
            "mlp_state":       self.mlp.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "feat_mean":       self.feat_mean,
            "feat_var":        self.feat_var,
            "feat_n":          self.feat_n,
            "sample_count":    self.sample_count,
            "perf_ema":        self.perf.ema,
            "perf_count":      self.perf.count,
        }
        torch.save(checkpoint, self.cfg.weights_path)
        print(f"[Selector] Saved → {self.cfg.weights_path}")

    def _load_weights(self) -> None:
        """
        Load a previously saved checkpoint if one exists.

        Silently starts fresh if the file is missing or incompatible.
        """
        if not os.path.exists(self.cfg.weights_path):
            return
        try:
            checkpoint = torch.load(
                self.cfg.weights_path, map_location=self.device, weights_only=False
            )
            self.mlp.load_state_dict(checkpoint["mlp_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.feat_mean    = checkpoint["feat_mean"]
            self.feat_var     = checkpoint["feat_var"]
            self.feat_n       = checkpoint["feat_n"]
            self.sample_count = checkpoint["sample_count"]
            if "perf_ema" in checkpoint:
                self.perf.ema   = checkpoint["perf_ema"]
                self.perf.count = checkpoint["perf_count"]
            print(
                f"[Selector] Loaded weights ({self.sample_count} prior samples) "
                f"from '{self.cfg.weights_path}'"
            )
        except Exception as exc:
            print(f"[Selector] Could not load weights: {exc}. Starting fresh.")
