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

def get_cosine_lr(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_pct: float = 0.05,   # proporção do aquecimento (ex.: 2000/max_steps)
    decay_pct:  float = 0.90,   # proporção do decaimento cosseno (ex.: 0.90)
    min_lr:    float = 1e-6
) -> float:
    """
    Cosine decay com warmup linear e cauda constante.
    - Warmup: LR sobe linearmente de 0 até base_lr em warmup_pct * max_steps.
    - Cosine: LR decai de base_lr até 10% de base_lr ao longo de decay_pct * max_steps.
    - Tail:   Após o decaimento, mantém LR constante nesse valor final.
    """

    # Particiona os passos
    warmup_steps  = int(max_steps * warmup_pct)
    cosine_steps  = int(max_steps * decay_pct)
    tail_steps    = max(max_steps - warmup_steps - cosine_steps, 0)

    # Limites dos intervalos
    warmup_end = warmup_steps
    cosine_end = warmup_end + cosine_steps
    # tail = [cosine_end, max_steps)

    # LR final alvo: 10% do pico
    final_lr_target = base_lr * 0.10

    if step < warmup_end:  # warmup linear
        # evita divisão por zero em casos extremos
        denom = max(warmup_steps, 1)
        lr = base_lr * (step / denom)

    elif step < cosine_end:  # cosine decay de base_lr -> final_lr_target
        t = step - warmup_end
        denom = max(cosine_steps, 1)
        p = min(max(t / denom, 0.0), 1.0)  # progresso em [0,1]
        # fórmula do cosine annealing: interpola de base_lr (p=0) até final_lr_target (p=1)
        lr = final_lr_target + 0.5 * (1.0 + math.cos(math.pi * p)) * (base_lr - final_lr_target)

    else:  # cauda constante no valor final
        lr = final_lr_target

    return max(lr, min_lr)