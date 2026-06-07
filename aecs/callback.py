"""HuggingFace Trainer callback for AECS."""

from __future__ import annotations

from .scheduler import EventControlScheduler


class AECSCallback:
    """Plugs AECS into a HuggingFace Trainer.

    Usage::

        from aecs import AECSConfig, EventControlScheduler
        from aecs.callback import AECSCallback

        trainer = Trainer(
            ...,
            optimizers=(optimizer, None),  # disable HF scheduler
        )
        aecs = EventControlScheduler(optimizer, AECSConfig(total_steps=10000))
        trainer.add_callback(AECSCallback(aecs))
    """

    def __init__(self, scheduler: EventControlScheduler):
        self.scheduler = scheduler
        self._last_loss: float = 0.0
        self._last_grad_norm: float = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        # HF Trainer only calls on_log every `logging_steps` (default: 500), so the
        # scheduler is eating stale loss/grad_norm for potentially hundreds of steps
        # between log events. If you want per-step fidelity, you're on your own.
        if logs:
            self._last_loss = float(logs.get("loss", self._last_loss))
            self._last_grad_norm = float(logs.get("grad_norm", self._last_grad_norm))

    def on_step_end(self, args, state, control, **kwargs):
        self.scheduler.step(
            {
                "loss": self._last_loss,
                "grad_norm": self._last_grad_norm,
            }
        )
