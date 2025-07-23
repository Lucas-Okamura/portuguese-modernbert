import time
import torch
import logging
from accelerate import Accelerator
from lr_scheduler import get_trapezoidal_lr
from checkpoint_utils import save_checkpoint
from stableadamw import StableAdamW

def train(
    args, model, tokenizer, dataloader, start_step
):
    accelerator = Accelerator(mixed_precision='fp16')
    optimizer = StableAdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        eps=1e-6,
        decouple_lr=False,
        return_norms=True
    )

    logging.info("Initializing model with Accelerator")
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    logging.info("Initializing training")
    model.train()
    optimizer.zero_grad()
    start_time = time.time()
    for step, batch in enumerate(dataloader, start=start_step):
        lr = get_trapezoidal_lr(
            step, args.warmup_steps,
            args.max_steps, args.lr
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

        if step % args.log_every == 0 and accelerator.is_main_process:
            elapsed = int(time.time() - start_time)
            hrs, rem = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            logging.info(
                f"Step {step} - Loss: {loss:.4f} - LR: {lr:.2e}"
                f" - Time: {hrs}h {mins}m {secs}s"
            )

        if step % args.save_every == 0 and step != 0 and accelerator.is_main_process:
            save_checkpoint(
                accelerator.unwrap_model(model),
                tokenizer,
                args.checkpoint_dir,
                step
            )
        if step >= args.max_steps:
            break

    if accelerator.is_main_process:
        save_checkpoint(
            accelerator.unwrap_model(model),
            tokenizer,
            args.output_dir,
            'final'
        )
        logging.info("Training complete.")
    
    accelerator.wait_for_everyone()
