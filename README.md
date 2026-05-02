# AECS — Adaptive Event-Control Scheduler

A state-aware, event-driven learning rate scheduler for PyTorch. Instead of following a fixed decay curve, AECS watches your training signals in real time and switches between four operating modes based on what it detects.

**Benchmarked #3 out of 16 schedulers** on SST-2/DistilBERT at [juiceb0xc0de/lr-scheduler-benchmark](https://huggingface.co/spaces/juiceb0xc0de/lr-scheduler-benchmark).

## How it works

AECS runs a cosine backbone and modulates on top using event-driven mode switching:

| Mode | Trigger | Effect |
|---|---|---|
| `BASELINE` | Default | Cosine backbone, normal settings |
| `RECOVERY` | Gradient spike or loss spike | LR × 0.3, momentum dampened |
| `EXPLORE` | High gradient redundancy or plateau | LR × 1.5, weight decay halved |
| `STABILIZE` | Persistent instability | LR × 0.24, extended dampening |

Events are detected via:
- **ZClip-style z-score** on gradient norm (spike detection)
- **Loss ratio** vs recent minimum (loss explosion)
- **Gradient cosine similarity** (redundancy / plateau)
- **Grad norm variance** (instability)

Hysteresis + cooldown periods prevent mode jitter.

## Install

```bash
pip install git+https://github.com/JuiceB0xC0de/aecs-scheduler.git
```

## Usage

```python
import torch
from aecs import AECSConfig, EventControlScheduler

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
```

## HuggingFace Trainer

```python
from aecs import AECSConfig, EventControlScheduler
from aecs.callback import AECSCallback

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
aecs = EventControlScheduler(optimizer, AECSConfig(total_steps=10000))

trainer = Trainer(
    ...,
    optimizers=(optimizer, None),  # disable HF's built-in scheduler
)
trainer.add_callback(AECSCallback(aecs))
```

## Config reference

```python
AECSConfig(
    base_lr=5e-5,
    warmup_steps=500,
    total_steps=10000,
    instability_z_thresh=2.5,    # z-score threshold for gradient spike
    loss_spike_ratio=1.15,       # loss > 1.15x recent min triggers RECOVERY
    redundancy_thresh=0.98,      # cosine similarity threshold for EXPLORE
    plateau_grad_norm_thresh=1e-4,
    recovery_lr_factor=0.3,
    explore_lr_factor=1.5,
    cooldown_steps=200,          # min steps before mode can switch again
    verbose=True,
)
```

## Benchmark results

| Rank | Scheduler | Val Loss | Val Acc |
|---|---|---|---|
| 1 | DLRS | 0.2653 | 89.03% |
| 2 | GreedyLR | 0.2688 | 89.22% |
| **3** | **AECS** | **0.2985** | **90.44%** |
| 4 | Cosine w/ Restarts | 0.3065 | 90.48% |
| 10 | Cosine (baseline) | 0.3533 | 90.33% |

Task: SST-2 · Model: DistilBERT-base · 3 seeds · [full leaderboard](https://huggingface.co/spaces/juiceb0xc0de/lr-scheduler-benchmark)

## License

MIT
