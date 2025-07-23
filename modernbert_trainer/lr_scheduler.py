def get_trapezoidal_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float
) -> float:
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    elif step < max_steps * 0.9:
        return base_lr
    decay_ratio = (step - max_steps * 0.9) / (max_steps * 0.1)
    return base_lr * (1 - decay_ratio ** 0.5)