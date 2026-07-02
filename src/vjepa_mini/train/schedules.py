"""Hyper-parameter schedules, exactly as V-JEPA uses them (App. C).

Three quantities change over training:
  lr        — warmup then cosine decay (standard for ViTs; large models
              diverge without warmup because early gradients are noisy);
  wd        — linearly INCREASED 0.04 -> 0.4 (mild regularization early
              while features form, strong late to stop overfitting);
  momentum  — EMA momentum linearly 0.996 -> 1.0 (targets update fast at
              first, then freeze — the curriculum goes from "easy, moving
              targets" to "stable, semantic targets").

The paper's odd trick: all schedules are computed for 1.25x the true
training length, then truncated. The tail of a cosine schedule changes
hyper-parameters too aggressively; cutting it off helped their results.
"""

import math


def cosine_with_warmup(
    step: int,
    total_steps: int,
    base_lr: float,
    final_lr: float = 0.0,
    warmup_steps: int = 0,
    schedule_scale: float = 1.0,
) -> float:
    """Linear warmup from 0, then cosine decay toward final_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    horizon = int(total_steps * schedule_scale)  # stretch, then truncate
    progress = (step - warmup_steps) / max(1, horizon - warmup_steps)
    progress = min(progress, 1.0)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))


def linear_schedule(
    step: int, total_steps: int, start: float, end: float, schedule_scale: float = 1.0
) -> float:
    """Linear interpolation start -> end over a (possibly stretched) horizon."""
    horizon = int(total_steps * schedule_scale)
    progress = min(step / max(1, horizon), 1.0)
    return start + (end - start) * progress
