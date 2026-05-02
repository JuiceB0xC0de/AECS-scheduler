"""
Adaptive Event-Control Scheduler (AECS)
A state-aware, event-driven training controller for PyTorch.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer


@dataclass
class AECSConfig:
    """Configuration for AECS."""
    base_lr: float = 5e-5
    warmup_steps: int = 500
    total_steps: int = 10000

    loss_window: int = 50
    grad_window: int = 50
    redundancy_window: int = 20

    instability_z_thresh: float = 2.5       # ZClip-style gradient anomaly
    loss_spike_ratio: float = 1.15          # loss > 1.15x min_recent_loss
    redundancy_thresh: float = 0.98         # cosine similarity of grads
    plateau_grad_norm_thresh: float = 1e-4  # near-zero gradient norm

    recovery_lr_factor: float = 0.3
    recovery_momentum_factor: float = 0.5
    recovery_clip_boost: float = 1.5
    recovery_min_steps: int = 100
    recovery_max_steps: int = 1000

    explore_lr_factor: float = 1.5
    explore_noise_std: float = 1e-5

    event_persistence: int = 5              # steps signal must persist
    cooldown_steps: int = 200               # min steps in a mode before switching
    reentry_grad_norm_tol: float = 0.1      # grad norm variance must be < this

    verbose: bool = True


class SignalBuffer:
    """Ring buffer for training signals with derived statistics."""

    def __init__(self, window: int = 50):
        self.losses: deque = deque(maxlen=window)
        self.grad_norms: deque = deque(maxlen=window)
        self.grad_cosines: deque = deque(maxlen=window)
        self.layer_grad_norms: deque = deque(maxlen=window)
        self.steps: int = 0
        self._prev_grad_flat: Optional[torch.Tensor] = None

    def push(self, loss: float, grad_norm: float, layer_grad_norms: Optional[List[float]] = None):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.layer_grad_norms.append(layer_grad_norms or [])
        self.steps += 1

    def push_grad_cosine(self, grad_flat: torch.Tensor):
        if self._prev_grad_flat is not None:
            cos = torch.nn.functional.cosine_similarity(
                grad_flat, self._prev_grad_flat, dim=0
            ).item()
            self.grad_cosines.append(cos)
        self._prev_grad_flat = grad_flat.clone()

    def loss_min_recent(self, n: int = 10) -> float:
        if len(self.losses) == 0:
            return float('inf')
        return min(list(self.losses)[-n:])

    def grad_norm_ema(self, alpha: float = 0.97) -> Tuple[float, float]:
        if len(self.grad_norms) == 0:
            return 0.0, 1.0
        mu = self.grad_norms[0]
        var = 0.0
        for g in list(self.grad_norms)[1:]:
            mu = alpha * mu + (1 - alpha) * g
            var = alpha * var + (1 - alpha) * (g - mu) ** 2
        return mu, math.sqrt(max(var, 1e-8))

    def grad_norm_zscore(self) -> float:
        if len(self.grad_norms) < 5:
            return 0.0
        mu, sigma = self.grad_norm_ema()
        if sigma < 1e-8:
            return 0.0
        return (self.grad_norms[-1] - mu) / sigma

    def grad_norm_variance(self) -> float:
        if len(self.grad_norms) < 5:
            return 0.0
        vals = list(self.grad_norms)
        mean = sum(vals) / len(vals)
        return sum((x - mean) ** 2 for x in vals) / len(vals)

    def redundancy_score(self) -> float:
        if len(self.grad_cosines) < max(1, self.grad_cosines.maxlen // 2):
            return 0.0
        vals = list(self.grad_cosines)
        return sum(vals) / len(vals)

    def instability_score(self) -> float:
        return self.grad_norm_zscore()


class EventControlScheduler:
    """
    Adaptive Event-Control Scheduler (AECS).

    Modes:
        BASELINE   : normal training, cosine backbone
        RECOVERY   : shock detected — dampen LR, boost clipping
        EXPLORE    : high redundancy — increase LR/noise to escape flat regions
        STABILIZE  : persistent instability — lower LR longer-term

    Usage::

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = EventControlScheduler(optimizer, AECSConfig(total_steps=10000))

        for batch in dataloader:
            loss = train_step(batch)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scheduler.step({
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
            })

            optimizer.step()
            optimizer.zero_grad()
    """

    MODES = ["BASELINE", "RECOVERY", "EXPLORE", "STABILIZE"]

    def __init__(self, optimizer: Optimizer, config: Optional[AECSConfig] = None):
        self.optimizer = optimizer
        self.config = config or AECSConfig()
        self.buffer = SignalBuffer(
            window=max(self.config.loss_window, self.config.grad_window)
        )

        self.mode: str = "BASELINE"
        self.mode_steps: int = 0
        self.total_steps: int = 0
        self.event_counter: Dict[str, int] = {m: 0 for m in self.MODES}
        self.transition_log: List[Dict] = []

        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.base_betas = [g.get("betas", (0.9, 0.999)) for g in optimizer.param_groups]
        self.base_weight_decays = [g.get("weight_decay", 0.0) for g in optimizer.param_groups]

        self._loss_ema: float = 0.0
        self._loss_ema_alpha: float = 0.95

    def step(self, signals: Dict[str, float]) -> str:
        """Advance the scheduler one step.

        Args:
            signals: dict with keys ``"loss"`` and ``"grad_norm"``.

        Returns:
            Current mode string.
        """
        loss = signals.get("loss", 0.0)
        grad_norm = signals.get("grad_norm", 0.0)
        layer_grad_norms = signals.get("layer_grad_norms", None)

        if self.total_steps == 0:
            self._loss_ema = loss
        else:
            self._loss_ema = (
                self._loss_ema_alpha * self._loss_ema
                + (1 - self._loss_ema_alpha) * loss
            )

        self.buffer.push(loss, grad_norm, layer_grad_norms)
        self.total_steps += 1
        self.mode_steps += 1

        event = self._detect_event(signals)
        if event:
            self._maybe_transition(event)

        lrs = self._compute_lrs()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

        self._apply_mode_tweaks()
        return self.mode

    def _detect_event(self, signals: Dict[str, float]) -> Optional[str]:
        cfg = self.config
        buf = self.buffer

        if buf.steps < cfg.event_persistence + 5:
            return None

        events = []

        z = buf.grad_norm_zscore()
        if z > cfg.instability_z_thresh:
            events.append("GRADIENT_SPIKE")

        recent_min = buf.loss_min_recent(n=10)
        if recent_min > 0 and buf.losses[-1] > recent_min * cfg.loss_spike_ratio:
            events.append("LOSS_SPIKE")

        redundancy = buf.redundancy_score()
        if redundancy > cfg.redundancy_thresh:
            events.append("REDUNDANT")

        grad_var = buf.grad_norm_variance()
        if grad_var > cfg.reentry_grad_norm_tol and z > 1.0:
            events.append("UNSTABLE")

        if len(buf.grad_norms) >= 10:
            recent_avg = sum(list(buf.grad_norms)[-10:]) / 10
            if recent_avg < cfg.plateau_grad_norm_thresh:
                events.append("PLATEAU")

        if not events:
            return None

        for p in ["GRADIENT_SPIKE", "LOSS_SPIKE", "UNSTABLE", "REDUNDANT", "PLATEAU"]:
            if p in events:
                return p
        return events[0]

    def _maybe_transition(self, event: str):
        cfg = self.config

        if self.mode_steps < cfg.cooldown_steps and self.mode != "BASELINE":
            return

        if self.mode == "RECOVERY":
            if self.mode_steps < cfg.recovery_min_steps:
                return
            if self.mode_steps >= cfg.recovery_max_steps:
                self._enter_mode("BASELINE", "recovery_max_duration")
                return

        target = self._event_to_mode(event)
        if target == self.mode:
            return

        if target == "BASELINE" and self.mode in ("RECOVERY", "STABILIZE"):
            if self.buffer.grad_norm_variance() > cfg.reentry_grad_norm_tol:
                return

        self._enter_mode(target, event)

    def _event_to_mode(self, event: str) -> str:
        return {
            "GRADIENT_SPIKE": "RECOVERY",
            "LOSS_SPIKE":     "RECOVERY",
            "UNSTABLE":       "STABILIZE",
            "REDUNDANT":      "EXPLORE",
            "PLATEAU":        "EXPLORE",
        }.get(event, "BASELINE")

    def _enter_mode(self, new_mode: str, cause: str):
        old_mode = self.mode
        self.mode = new_mode
        self.mode_steps = 0
        self.event_counter[new_mode] += 1
        self.transition_log.append({
            "step": self.total_steps,
            "from": old_mode,
            "to": new_mode,
            "cause": cause,
            "lr": self.optimizer.param_groups[0]["lr"],
        })
        if self.config.verbose:
            print(f"[AECS] Step {self.total_steps}: {old_mode} -> {new_mode} ({cause})")

    def _compute_lrs(self) -> List[float]:
        cfg = self.config
        step = self.total_steps

        if step < cfg.warmup_steps:
            backbone = step / max(cfg.warmup_steps, 1)
        else:
            progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
            backbone = 0.5 * (1.0 + math.cos(math.pi * progress))

        mult = 1.0
        if self.mode == "RECOVERY":
            mult = cfg.recovery_lr_factor
        elif self.mode == "EXPLORE":
            mult = cfg.explore_lr_factor
        elif self.mode == "STABILIZE":
            mult = cfg.recovery_lr_factor * 0.8

        return [base * backbone * mult for base in self.base_lrs]

    def _apply_mode_tweaks(self):
        cfg = self.config
        for group, base_b, base_wd in zip(
            self.optimizer.param_groups, self.base_betas, self.base_weight_decays
        ):
            if "betas" in group:
                beta1, beta2 = base_b
                if self.mode == "RECOVERY":
                    group["betas"] = (beta1 * cfg.recovery_momentum_factor, beta2)
                else:
                    group["betas"] = (beta1, beta2)
            if "weight_decay" in group:
                if self.mode == "EXPLORE":
                    group["weight_decay"] = base_wd * 0.5
                elif self.mode == "RECOVERY":
                    group["weight_decay"] = base_wd * 1.2
                else:
                    group["weight_decay"] = base_wd

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def summary(self) -> Dict:
        return {
            "mode": self.mode,
            "total_steps": self.total_steps,
            "transitions": len(self.transition_log),
            "event_counter": self.event_counter,
            "recent_transitions": self.transition_log[-5:],
        }
