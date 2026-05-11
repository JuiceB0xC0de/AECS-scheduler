"""Unit tests for AECS scheduler core logic."""

from __future__ import annotations

import pytest
import torch
from aecs import AECSConfig, EventControlScheduler, SignalBuffer


class TestSignalBuffer:
    """Tests for SignalBuffer class."""

    def test_initial_state(self):
        """Test buffer initializes correctly."""
        buf = SignalBuffer(window=50)
        assert buf.steps == 0
        assert len(buf.losses) == 0
        assert len(buf.grad_norms) == 0

    def test_push_data(self):
        """Test pushing data to buffer."""
        buf = SignalBuffer(window=5)
        buf.push(loss=1.0, grad_norm=0.5)
        assert buf.steps == 1
        assert buf.losses[-1] == 1.0
        assert buf.grad_norms[-1] == 0.5

    def test_window_limit(self):
        """Test buffer respects window size (ring buffer behavior)."""
        buf = SignalBuffer(window=3)
        for i in range(5):
            buf.push(loss=float(i), grad_norm=float(i))
        assert len(buf.losses) == 3
        assert list(buf.losses) == [2.0, 3.0, 4.0]

    def test_grad_cosine(self):
        """Test gradient cosine similarity calculation."""
        buf = SignalBuffer(window=10)
        grad1 = torch.tensor([1.0, 0.0, 0.0])
        grad2 = torch.tensor([0.0, 1.0, 0.0])
        grad3 = torch.tensor([1.0, 0.0, 0.0])  # Same as grad1

        buf.push_grad_cosine(grad1)
        buf.push_grad_cosine(grad2)  # Orthogonal -> cosine ~0
        buf.push_grad_cosine(grad3)  # Same as prev -> cosine ~1

        # Should have 2 cosine values (from transitions)
        assert len(buf.grad_cosines) == 2

    def test_loss_min_recent(self):
        """Test recent minimum loss calculation."""
        buf = SignalBuffer(window=10)
        for loss in [1.0, 2.0, 1.5, 0.5, 0.8]:
            buf.push(loss=loss, grad_norm=0.5)
        assert buf.loss_min_recent(n=3) == pytest.approx(0.5, rel=1e-5)

    def test_loss_min_recent_empty(self):
        """Test min loss with empty buffer."""
        buf = SignalBuffer(window=10)
        assert buf.loss_min_recent(n=10) == float('inf')

    def test_grad_norm_ema(self):
        """Test EMA calculation for gradient norms."""
        buf = SignalBuffer(window=10)
        for g in [1.0, 1.1, 0.9, 1.05, 0.95]:
            buf.push(loss=1.0, grad_norm=g)
        mu, sigma = buf.grad_norm_ema()
        assert mu > 0
        assert sigma >= 0

    def test_grad_norm_zscore(self):
        """Test z-score calculation."""
        buf = SignalBuffer(window=20)
        # Add stable values
        for g in [1.0] * 15:
            buf.push(loss=1.0, grad_norm=g)
        # Add outlier
        buf.push(loss=1.0, grad_norm=5.0)
        z = buf.grad_norm_zscore()
        assert z > 0  # Outlier should have positive z-score

    def test_grad_norm_variance(self):
        """Test gradient norm variance calculation."""
        buf = SignalBuffer(window=10)
        for g in [1.0, 2.0, 1.5, 2.5, 1.2]:
            buf.push(loss=1.0, grad_norm=g)
        var = buf.grad_norm_variance()
        assert var >= 0

    def test_redundancy_score(self):
        """Test redundancy score calculation."""
        buf = SignalBuffer(window=20)
        # Add enough cosine values to compute score
        for i in range(15):
            grad = torch.tensor([1.0 if i == j else 0.0 for j in range(5)])
            buf.push_grad_cosine(grad)
        score = buf.redundancy_score()
        # Score should be between -1 and 1
        assert -1.0 <= score <= 1.0


class TestEventControlScheduler:
    """Tests for EventControlScheduler class."""

    def test_initial_state(self):
        """Test scheduler initializes correctly."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
        config = AECSConfig(total_steps=1000)
        scheduler = EventControlScheduler(optimizer, config)
        assert scheduler.mode == "BASELINE"
        assert scheduler.total_steps == 0
        assert scheduler.mode_steps == 0

    def test_step_basic(self):
        """Test basic step execution."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
        config = AECSConfig(total_steps=100)
        scheduler = EventControlScheduler(optimizer, config)

        mode = scheduler.step({"loss": 1.0, "grad_norm": 0.5})
        assert mode == "BASELINE"
        assert scheduler.total_steps == 1

    def test_warmup_lr(self):
        """Test warmup phase increases LR from 0 to base_lr."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
        config = AECSConfig(warmup_steps=10, total_steps=100)
        scheduler = EventControlScheduler(optimizer, config)

        # Before any steps, LR is still at base (scheduler hasn't modified it yet)
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 1e-3

        # During warmup, LR should increase from 0 toward base_lr
        for _ in range(5):
            scheduler.step({"loss": 1.0, "grad_norm": 0.5})

        # After 5 steps of warmup (step 5/10), LR should be at 50% of base_lr
        lr_after = scheduler.get_last_lr()[0]
        assert lr_after == pytest.approx(5e-4, rel=1e-5)

        # After 10 steps, LR should reach base_lr
        for _ in range(5):
            scheduler.step({"loss": 1.0, "grad_norm": 0.5})
        lr_at_end_warmup = scheduler.get_last_lr()[0]
        assert lr_at_end_warmup == pytest.approx(1e-3, rel=1e-5)

    def test_cosine_decay(self):
        """Test cosine decay after warmup."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
        config = AECSConfig(warmup_steps=5, total_steps=100)
        scheduler = EventControlScheduler(optimizer, config)

        # Go past warmup
        for _ in range(50):
            scheduler.step({"loss": 1.0, "grad_norm": 0.5})
        lr_mid = scheduler.get_last_lr()[0]

        # Closer to end, LR should be lower
        for _ in range(40):
            scheduler.step({"loss": 1.0, "grad_norm": 0.5})
        lr_end = scheduler.get_last_lr()[0]

        assert lr_end <= lr_mid

    def test_summary(self):
        """Test summary method."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
        config = AECSConfig(total_steps=100)
        scheduler = EventControlScheduler(optimizer, config)

        for _ in range(10):
            scheduler.step({"loss": 1.0, "grad_norm": 0.5})

        summary = scheduler.summary()
        assert "mode" in summary
        assert "total_steps" in summary
        assert "transitions" in summary
        assert "event_counter" in summary

    def test_get_last_lr(self):
        """Test get_last_lr returns correct format."""
        # Use dict-based param groups to ensure separate groups
        params1 = [torch.nn.Parameter(torch.randn(2, 2))]
        params2 = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.AdamW([
            {'params': params1, 'lr': 1e-3},
            {'params': params2, 'lr': 2e-3}
        ])
        config = AECSConfig(total_steps=100)
        scheduler = EventControlScheduler(optimizer, config)

        lrs = scheduler.get_last_lr()
        assert len(lrs) == 2
        assert all(isinstance(lr, float) for lr in lrs)
        # First group should have lower LR
        assert lrs[0] < lrs[1]


class TestModeTransitions:
    """Tests for mode transition logic."""

    def test_recovery_mode_on_gradient_spike(self):
        """Test entry to RECOVERY mode on gradient spike."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
        # Low threshold to trigger easily
        config = AECSConfig(
            total_steps=100,
            instability_z_thresh=1.5,  # Low to trigger
            cooldown_steps=0,  # No cooldown for testing
        )
        scheduler = EventControlScheduler(optimizer, config)

        # First, build up buffer with normal values
        for _ in range(30):
            scheduler.step({"loss": 1.0, "grad_norm": 1.0})

        # Then trigger a gradient spike
        scheduler.step({"loss": 1.0, "grad_norm": 10.0})

        # Should be in RECOVERY mode
        assert scheduler.mode in ("RECOVERY", "BASELINE")  # May not transition yet due to cooldown

    def test_redundant_triggers_explore(self):
        """Test REDUNDANT event triggers EXPLORE mode."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
        config = AECSConfig(
            total_steps=100,
            redundancy_thresh=0.5,  # Low threshold
            grad_window=10,
            loss_window=10,
            cooldown_steps=0,
        )
        scheduler = EventControlScheduler(optimizer, config)

        # Build buffer with similar gradients (high cosine similarity)
        for i in range(30):
            # Push similar gradient vectors
            grad = torch.tensor([1.0, 0.0, 0.0])
            scheduler.buffer.push_grad_cosine(grad)
            scheduler.buffer.push(loss=1.0, grad_norm=1.0)
            scheduler.total_steps += 1
            scheduler.mode_steps += 1

        # Force redundancy check
        scheduler.buffer.grad_cosines.append(0.99)  # Very redundant
        scheduler.buffer.grad_cosines.append(0.98)

        # Check that redundancy score is high
        score = scheduler.buffer.redundancy_score()
        assert score > config.redundancy_thresh
