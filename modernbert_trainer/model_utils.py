from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForMaskedLM
)
import logging

def load_config(args, resume_path=None):
    if resume_path:
        logging.info(f"Loading Config from {resume_path} with rope_theta={args.rope_theta}")
        cfg = AutoConfig.from_pretrained(resume_path)
    else:
        logging.info(f"Loading Config from {args.model_name} with rope_theta={args.rope_theta}")
        cfg = AutoConfig.from_pretrained(args.model_name)
    cfg.rope_theta = args.rope_theta
    cfg.hidden_act = 'gelu_new'
    return cfg


def init_model_and_tokenizer(
    args, cfg, resume_path=None
):
    if resume_path:
        logging.info(f"Resuming from checkpoint: {resume_path}")
        tokenizer = AutoTokenizer.from_pretrained(resume_path)
        model = AutoModelForMaskedLM.from_pretrained(
            resume_path, config=cfg
        )
    else:
        logging.info(f"Starting training from scratch of model {args.model_name}.")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            model_max_length=args.max_length
        )
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name, config=cfg
        )
    # align special tokens
    for tok in ['pad', 'cls', 'sep', 'mask']:
        setattr(
            model.config,
            f"{tok}_token_id",
            getattr(tokenizer, f"{tok}_token_id")
        )
    # resize embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer