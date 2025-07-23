import os
from typing import Tuple, Optional


def get_latest_checkpoint(
    checkpoint_dir: str
) -> Tuple[Optional[str], int]:
    os.makedirs(checkpoint_dir, exist_ok=True)
    cks = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt-")]
    if not cks:
        return None, 0
    steps = sorted(int(f.split("-")[-1]) for f in cks)
    last = steps[-1]
    return os.path.join(checkpoint_dir, f"ckpt-{last}"), last


def save_checkpoint(
    model, tokenizer, checkpoint_dir: str, step: int
):
    save_path = os.path.join(checkpoint_dir, f"ckpt-{step}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(
        os.path.join(checkpoint_dir, "last_checkpoint.txt"), 'w'
    ) as f:
        f.write(save_path)