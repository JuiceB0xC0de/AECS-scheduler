# AGENTS.md — AECS-scheduler

## What this repo is

AECS (Adaptive Event-Control Scheduler) is a state-aware, event-driven learning rate
scheduler for PyTorch. Instead of following a fixed decay curve, it watches training signals
in real time and switches between four operating modes based on detected events. It ranked
**3rd out of 25 schedulers** on SST-2/DistilBERT at the
[lr-scheduler-benchmark](https://huggingface.co/juiceb0xc0de/lr-scheduler-benchmark).

Published on PyPI as `aecs-scheduler`. Install:
```bash
pip install git+https://github.com/JuiceB0xC0de/aecs-scheduler.git
```

## Repo layout

```
aecs/
  __init__.py       # exports AECSConfig, EventControlScheduler
  scheduler.py      # core: AECSConfig, SignalBuffer, EventControlScheduler
  callback.py       # HuggingFace Trainer integration (AECSCallback)
tests/
  test_scheduler.py
```

All logic lives in `aecs/scheduler.py`. Keep it that way — no splitting across files without
good reason.

## How AECS works

AECS runs a **cosine LR backbone** and modulates on top via event-driven mode switching.

### Modes

| Mode | Trigger event | LR factor | Other tweaks |
|------|--------------|-----------|--------------|
| `BASELINE` | Default | 1.0× | Normal betas, weight decay |
| `RECOVERY` | Gradient spike or loss spike | 0.3× | beta1 dampened, weight decay ×1.2 |
| `EXPLORE` | High gradient redundancy or plateau | 1.5× | weight decay ×0.5 |
| `STABILIZE` | Persistent instability | 0.24× | Extended dampening |

### Event detection (in `_detect_event`)

Events are detected in priority order: `GRADIENT_SPIKE` > `LOSS_SPIKE` > `UNSTABLE` >
`REDUNDANT` > `PLATEAU`.

- **GRADIENT_SPIKE** — ZClip-style z-score on gradient norm exceeds `instability_z_thresh`
- **LOSS_SPIKE** — current loss > `loss_spike_ratio` × recent minimum loss
- **REDUNDANT** — mean gradient cosine similarity exceeds `redundancy_thresh`
- **UNSTABLE** — grad norm variance > `reentry_grad_norm_tol` and z-score > 1.0
- **PLATEAU** — 10-step average grad norm < `plateau_grad_norm_thresh`

### Mode transitions (`_maybe_transition`)

- Transitions are gated by `cooldown_steps` — the scheduler won't switch modes until it has
  been in the current mode for at least that many steps (except from `BASELINE`).
- `RECOVERY` has its own `recovery_min_steps` / `recovery_max_steps` guards.
- Re-entry to `BASELINE` from `RECOVERY` or `STABILIZE` requires grad norm variance to drop
  below `reentry_grad_norm_tol`.

### `SignalBuffer`

Ring buffer (deque) for loss and grad norm history. Computes EMA-based z-scores, cosine
similarity between consecutive gradient flats, and redundancy scores. Warm-up period: event
detection is suppressed for the first `event_persistence + 5` steps.

## Key classes and their contracts

### `AECSConfig`
Frozen-ish dataclass. All thresholds, factors, and window sizes live here. Default values
are tuned for fine-tuning transformer models at `lr=5e-5`. See `scheduler.py` for field
docs.

### `EventControlScheduler`
- `__init__(optimizer, config)` — captures base LRs, betas, weight decays from param groups.
- `step(signals: dict) -> str` — must be called **before** `optimizer.step()`. Accepts
  `"loss"`, `"grad_norm"`, and optionally `"layer_grad_norms"`. Returns current mode string.
- `get_last_lr()` — returns list of current LRs per param group.
- `summary()` — returns dict with mode, step count, transition log (last 5), event counters.
- `transition_log` — full list of mode transitions for post-hoc analysis.

### `AECSCallback` (`aecs/callback.py`)
HuggingFace `TrainerCallback` that calls `scheduler.step()` at the right point in the
Trainer loop. Pass it via `trainer.add_callback(AECSCallback(scheduler))` and pass
`optimizers=(optimizer, None)` to disable HF's built-in scheduler.

## Usage pattern

```python
import torch
from aecs import AECSConfig, EventControlScheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = EventControlScheduler(optimizer, AECSConfig(total_steps=10000))

for batch in dataloader:
    loss = train_step(batch)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scheduler.step({"loss": loss.item(), "grad_norm": grad_norm.item()})

    optimizer.step()
    optimizer.zero_grad()
```

**Critical ordering:** `scheduler.step()` → `optimizer.step()` → `optimizer.zero_grad()`.
Swapping step order will apply the wrong LR for that batch.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

Tests live in `tests/test_scheduler.py`. When adding features:
- Test that the new event or mode transition fires under the expected signal conditions.
- Test that cooldown/hysteresis prevents spurious mode jitter.
- Keep tests runnable without a GPU — `torch.optim.AdamW` on CPU tensors is fine.

Code style: `black` (line length 88) + `ruff`. Run before committing:
```bash
black aecs/ tests/
ruff check aecs/ tests/
```

## Conventions

- **Python ≥ 3.9.** `torch >= 2.0` is the only runtime dependency for core code.
  `transformers >= 4.30` is optional (`[hf]` extra) for the callback.
- All signal history lives in `SignalBuffer`, not on the scheduler itself.
- Mode logic (detection, transition, LR computation, tweaks) stays in `scheduler.py`.
  The callback is thin — it should never contain scheduling logic.
- `transition_log` entries are dicts with keys `step`, `from`, `to`, `cause`, `lr`.
  Do not change this schema without updating downstream consumers.

## Related repos

- [event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer) —
  uses AECS to drive the Augmented-Lagrangian SAE training loop.
- [lr-scheduler-benchmark](https://huggingface.co/juiceb0xc0de/lr-scheduler-benchmark) —
  benchmark where AECS placed 3rd.
