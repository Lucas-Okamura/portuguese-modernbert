import math

def get_trapezoidal_lr(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_pct: float = 0.05,
    decay_pct:  float = 0.10,
    min_lr:    float = 1e-6
) -> float:
    import math
    warmup_steps  = int(max_steps * warmup_pct)
    decay_steps   = int(max_steps * decay_pct)
    plateau_steps = max(max_steps - warmup_steps - decay_steps, 0)

    if step < warmup_steps:                       # warm-up
        lr = base_lr * step / warmup_steps
    elif step < warmup_steps + plateau_steps:     # plateau
        lr = base_lr
    else:                                         # decay 1−√p
        p = (step - warmup_steps - plateau_steps) / max(decay_steps, 1)
        p = min(max(p, 0.0), 1.0)
        lr = base_lr * (1.0 - math.sqrt(p))

    return max(lr, min_lr)  