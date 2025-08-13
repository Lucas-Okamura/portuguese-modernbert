import os
import logging
from config import parse_args
from logging_utils import setup_logging
from model_utils import load_config, init_model_and_tokenizer
from data import get_dataloader
from checkpoint_utils import get_latest_checkpoint
from trainer import train


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    setup_logging(args.log_dir)
    for arg in vars(args):
        logging.info(f"Training with args: {arg}: {getattr(args, arg)}")

    resume_path, start_step = get_latest_checkpoint(
        args.checkpoint_dir
    ) if not args.resume_from else (args.resume_from,
                                     int(args.resume_from.split('-')[-1]))

    cfg = load_config(args, resume_path=resume_path)
    
    model, tokenizer = init_model_and_tokenizer(
        args, cfg, resume_path=resume_path
    )

    logging.info("Initializing Data Loader")
    dataloader, total_samples = get_dataloader(
        args, tokenizer, start_step
    )

    logging.info(f"Starting training from step {start_step}")
    train(args, model, tokenizer, dataloader, start_step, total_samples)


if __name__ == "__main__":
    main()
