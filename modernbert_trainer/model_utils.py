import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForMaskedLM
)
import logging

def load_config(args, resume_path=None):
    if resume_path:
        logging.info(f"Loading Config from {resume_path} with rope_theta={args.rope_theta}")
        cfg = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
    else:
        logging.info(f"Loading Config from {args.model_name} with rope_theta={args.rope_theta}")
        cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    cfg.rope_theta = args.rope_theta
    return cfg

def init_model_and_tokenizer(
    args, cfg, resume_path=None
):
    if resume_path:
        logging.info(f"Resuming from checkpoint: {resume_path}")
        tokenizer = AutoTokenizer.from_pretrained(resume_path)
        model = AutoModelForMaskedLM.from_pretrained(
            resume_path,
            config=cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        logging.info(f"Starting training from scratch of model {args.model_name}.")

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            model_max_length=args.max_length,
            trust_remote_code=True
        )

        special = {}
        if tokenizer.pad_token  is None: special["pad_token"]  = "[PAD]"
        if tokenizer.cls_token  is None: special["cls_token"]  = "[CLS]"
        if tokenizer.sep_token  is None: special["sep_token"]  = "[SEP]"
        if tokenizer.mask_token is None: special["mask_token"] = "[MASK]"
        if tokenizer.unk_token is None: special["unk_token"] = "[UNK]"
        if special:
            tokenizer.add_special_tokens(special)

        # Adapt model config to new tokenizer
        cfg.vocab_size = len(tokenizer)
        cfg.pad_token_id  = tokenizer.pad_token_id
        cfg.cls_token_id  = tokenizer.cls_token_id
        cfg.sep_token_id  = tokenizer.sep_token_id
        cfg.mask_token_id = tokenizer.mask_token_id
        cfg.unk_token_id = tokenizer.unk_token_id

        # Load Model
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            config=cfg,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )

    return model, tokenizer