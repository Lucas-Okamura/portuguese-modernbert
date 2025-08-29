import math
import numpy as np
import time
import torch
import logging
from accelerate import Accelerator
from lr_scheduler import get_trapezoidal_lr, get_cosine_lr
from checkpoint_utils import save_checkpoint
from stableadamw import StableAdamW
from torch.optim import AdamW

# --- NOVO: utilitário simples de EMA ---
class EMAMeter:
    def __init__(self, beta=0.98):
        self.beta = beta
        self.value_ = None
    def update(self, x: float) -> float:
        if self.value_ is None:
            self.value_ = x
        else:
            self.value_ = self.beta * self.value_ + (1 - self.beta) * x
        return self.value_
    @property
    def value(self):
        return float('inf') if self.value_ is None else float(self.value_)

def train(
    args,
    model,
    tokenizer,
    dataloader,
    start_step,
    total_samples=None
):
    accelerator = Accelerator(mixed_precision='bf16')

    if args.optimizer == "stableadamw":
        optimizer = StableAdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            weight_decay=1e-5,
            eps=1e-6,
            decouple_lr=False,
            return_norms=True
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-06,
            weight_decay=1e-5
        )

    logging.info("Initializing model with Accelerator")
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    logging.info("Initializing training")
    model.train()
    optimizer.zero_grad()
    start_time = time.time()

    num_samples = total_samples or args.total_samples
    num_gpus = accelerator.num_processes
    batch_size_per_gpu = dataloader.batch_size
    batch_size_total = batch_size_per_gpu * num_gpus
    steps_per_epoch = math.ceil(num_samples / batch_size_total)
    total_steps = steps_per_epoch * args.epochs

    logging.info(f"Dataset samples for training: {num_samples}")
    logging.info(f"No. GPUs: {num_gpus}")
    logging.info(f"Batch Size per GPU: {batch_size_per_gpu}")
    logging.info(f"Steps in dataset: {steps_per_epoch}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Total Steps: {total_steps}")
    if start_step > 0:
        logging.info(f"Resuming from global step {start_step}")

    # Cálculo de retomada
    start_epoch = (start_step // steps_per_epoch) + 1
    resume_in_epoch = start_step % steps_per_epoch
    remaining_steps = max(total_steps - start_step, 0)

    # --- NOVO: estado para "melhor modelo" ---
    ema = EMAMeter(beta=getattr(args, "ema_beta", 0.98))
    best_loss = float('inf')
    best_step = -1

    for epoch in range(1, args.epochs + 1):
        if epoch < start_epoch:
            continue
        for step_in_epoch, batch in enumerate(dataloader):
            if epoch == start_epoch and step_in_epoch < resume_in_epoch:
                continue

            # Passo global 0-indexed
            global_step = (epoch - 1) * steps_per_epoch + step_in_epoch

            # LR schedule
            if args.decay_type == "trapezoidal":
                lr = get_trapezoidal_lr(
                    global_step, total_steps, args.lr, args.warmup_pct, args.decay_pct, args.min_lr
                )
            elif args.decay_type == "cosine":
                lr = get_cosine_lr(
                    global_step, total_steps, args.lr, args.warmup_pct, args.decay_pct, args.min_lr
                )
            for g in optimizer.param_groups:
                g['lr'] = lr

            # Forward + loss
            outputs = model(**batch)
            loss = outputs.loss  # loss do micro-step (antes de dividir por grad_accum)
            loss_grad = loss / args.grad_accum
            accelerator.backward(loss_grad)

            # --- NOVO: computa loss médio entre processos para logging/critério ---
            with torch.no_grad():
                # Nota: usamos o loss do micro-step (antes de acumular) para medir progresso
                gathered = accelerator.gather_for_metrics(loss.detach())
                step_loss = gathered.mean().item()
                ema_loss = ema.update(step_loss)

            # Otimização quando fecha uma acumulação
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                # --- NOVO: salvar se melhor (usar EMA para robustez) ---
                if accelerator.is_main_process:
                    if ema_loss < best_loss:
                        best_loss = ema_loss
                        best_step = global_step + 1
                        save_checkpoint(
                            accelerator.unwrap_model(model),
                            tokenizer,
                            args.checkpoint_dir,
                            "best"  # sufixo/identificador especial
                        )
                        logging.info(
                            f"[BEST] step={best_step} ema_loss={best_loss:.6f} (salvo em 'best')"
                        )

            # Logging periódico
            if ((global_step + 1) % args.log_every == 0) and accelerator.is_main_process:
                elapsed = int(time.time() - start_time)
                hrs, rem = divmod(elapsed, 3600)
                mins, secs = divmod(rem, 60)
                train_percentage = np.round(100 * (global_step + 1) / total_steps, 2)
                logging.info(
                    f"Epoch {epoch} - Step {global_step + 1}/{total_steps} ({train_percentage} %) "
                    f"- Loss(step) {step_loss:.5f} - EMA {ema_loss:.5f} - LR {lr:.2e} "
                    f"- Time: {hrs}h {mins}m {secs}s"
                )

            # Checkpoint periódico (opcional)
            if ((global_step + 1) % args.save_every == 0) and global_step != 0 and accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(model),
                    tokenizer,
                    args.checkpoint_dir,
                    global_step + 1
                )

        # Save ao fim da época (como antes)
        if accelerator.is_main_process:
            end_of_epoch_step = min((epoch * steps_per_epoch), total_steps)
            save_checkpoint(
                accelerator.unwrap_model(model),
                tokenizer,
                args.checkpoint_dir,
                end_of_epoch_step
            )
            logging.info(f"Training complete for epoch {epoch}.")

    if accelerator.is_main_process:
        logging.info(f"Melhor modelo: step={best_step} ema_loss={best_loss:.6f} salvo como 'best'.")

    accelerator.wait_for_everyone()
