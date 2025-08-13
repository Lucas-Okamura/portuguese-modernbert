import math
import numpy as np
import time
import torch
import logging
from accelerate import Accelerator
from lr_scheduler import get_trapezoidal_lr
from checkpoint_utils import save_checkpoint
from stableadamw import StableAdamW
from torch.optim import AdamW

def train(
    args,
    model,
    tokenizer,
    dataloader,
    start_step,
    total_samples=None
):
    accelerator = Accelerator(mixed_precision='fp16')

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
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-06,
            weight_decay=1e-5
            )

    logging.info("Initializing model with Accelerator")
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

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

    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(dataloader, start=start_step):
            lr = get_trapezoidal_lr(
                step,
                total_steps,
                args.lr,
                args.warmup_pct,
                args.decay_pct,
                args.min_lr
            )
            for g in optimizer.param_groups:
                g['lr'] = lr
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            accelerator.backward(loss)
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if (step + 1) % args.log_every == 0 and accelerator.is_main_process:
                elapsed = int(time.time() - start_time)
                hrs, rem = divmod(elapsed, 3600)
                mins, secs = divmod(rem, 60)
                train_percentage = np.round(100 * step / total_steps, 2)
                logging.info(
                    f"Epoch {epoch} - Step {step + 1}/{total_steps} ({train_percentage} %) - Loss: {loss:.5f} - LR: {lr:.2e}"
                    f" - Time: {hrs}h {mins}m {secs}s"
                )

            if (step + 1) % args.save_every == 0 and step != 0 and accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(model),
                    tokenizer,
                    args.checkpoint_dir,
                    step
                )

        if accelerator.is_main_process:
            save_checkpoint(
                accelerator.unwrap_model(model),
                tokenizer,
                args.output_dir,
                f'final-epoch-{epoch}'
            )
            logging.info(f"Training complete for epoch {epoch}.")
    
    accelerator.wait_for_everyone()
