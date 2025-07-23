# train_modernbert_pt.py

import time
import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, AutoConfig
)
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
from stableadamw import StableAdamW
import torch
from datetime import datetime
import logging
from itertools import islice


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT on Portuguese Aroeira dataset")
    parser.add_argument("--output_dir", type=str, default="modernbert-pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--tokenizer_name", type=str, default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--dataset_name", type=str, default="Itau-Unibanco/aroeira")
    parser.add_argument("--train_part", type=str, default="pt1")
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mlm_probability", type=float, default=0.3)
    parser.add_argument("--grad_accum", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)


def get_trapezoidal_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    elif step < max_steps * 0.9:
        return base_lr
    else:
        decay_ratio = (step - max_steps * 0.9) / (max_steps * 0.1)
        return base_lr * (1 - decay_ratio ** 0.5)

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("ckpt-")]
    if not checkpoints:
        return None, 0
    steps = sorted([int(ckpt.split("-")[-1]) for ckpt in checkpoints])
    latest_step = steps[-1]
    return os.path.join(checkpoint_dir, f"ckpt-{latest_step}"), latest_step


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    setup_logging(args.log_dir)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    accelerator = Accelerator(mixed_precision="fp16")

    logging.info(f"Loading Tokenizer {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        model_max_length=args.max_length
    )

    resume_path, start_step = None, 0
    if args.resume_from:
        resume_path = args.resume_from
        start_step = int(resume_path.strip('/').split('-')[-1])
    else:
        resume_path, start_step = get_latest_checkpoint(args.checkpoint_dir)
    
    if resume_path:
        logging.info(f"Loading Config from {resume_path} with rope_theta={args.rope_theta}")
        cfg = AutoConfig.from_pretrained(resume_path)
    else:
        logging.info(f"Loading Config from {args.model_name} with rope_theta={args.rope_theta}")
        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.vocab_size = len(tokenizer)

    cfg.rope_theta = args.rope_theta
    cfg.hidden_act = "gelu_new"

    if resume_path:
        logging.info(f"Resuming from checkpoint: {resume_path} (step {start_step})")
        model = AutoModelForMaskedLM.from_pretrained(resume_path, config=cfg)
    else:
        logging.info(f"Starting training from scratch of model {args.model_name}.")
        model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=cfg)
    
    # Atribui dinamicamente no config do modelo
    model.config.pad_token_id  = tokenizer.pad_token_id
    model.config.cls_token_id  = tokenizer.cls_token_id
    model.config.sep_token_id  = tokenizer.sep_token_id
    model.config.mask_token_id = tokenizer.mask_token_id

    # if model.get_input_embeddings().num_embeddings != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))

    full_dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    if args.train_part == "pt1":
        sliced_dataset = islice(full_dataset, 0, 30_000_000)
    else:
        sliced_dataset = islice(full_dataset, 30_000_000, None)


    def skip_n(dataset, n):
        for idx, item in enumerate(dataset):
            if idx >= n:
                yield item

    skipped_dataset = skip_n(sliced_dataset, start_step * args.batch_size)

    def tokenize_function(example, step):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )

    def stream_tokenizer(data_stream, start_step):
        for i, example in enumerate(data_stream):
            step = i + start_step
            yield tokenize_function(example, step)

    tokenized_dataset = stream_tokenizer(skipped_dataset, start_step)

    class TokenizedIterableDataset(IterableDataset):
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            return iter(self.data)

    iterable_dataset = TokenizedIterableDataset(tokenized_dataset)

    logging.info("Initializing Data Collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability
    )

    logging.info("Initializing Data Loader")
    dataloader = DataLoader(iterable_dataset, batch_size=args.batch_size, collate_fn=data_collator)

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
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    logging.info("Initializing training")
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader, start=start_step):
        print(step)
        print(accelerator.is_main_process)
        print(step % args.save_every)
        print()
        lr = get_trapezoidal_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum
        accelerator.backward(loss)

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if step % args.log_every == 0 and accelerator.is_main_process:
            elapsed = int(time.time() - start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours}h {minutes}min {seconds}s"
            log_line = f"Step {step} - Loss: {loss.item():.4f} - LR: {lr:.2e} - Time: {time_str}"
            logging.info(log_line)

        if step % args.save_every == 0 and step != 0 and accelerator.is_main_process:
            model_to_save = accelerator.unwrap_model(model)
            save_path = f"{args.checkpoint_dir}/ckpt-{step}"
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            with open(os.path.join(args.checkpoint_dir, "last_checkpoint.txt"), "w") as f:
                f.write(save_path)

        if step >= args.max_steps:
            break
        
    if accelerator.is_main_process:
        final_model = accelerator.unwrap_model(model)
        final_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logging.info("Training complete.")
    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()