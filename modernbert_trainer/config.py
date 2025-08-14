import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ModernBERT on Portuguese Aroeira dataset"
    )
    parser.add_argument("--output_dir", type=str, default="modernbert-pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--model_name", type=str, default="answerdotai/ModernBERT-base"
    )
    parser.add_argument(
        "--tokenizer_name", type=str,
        default="ricardoz/BERTugues-base-portuguese-cased"
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="Itau-Unibanco/aroeira"
    )
    parser.add_argument("--optimizer", type=str, default="stableadamw")
    parser.add_argument("--train_part", type=str, default="pt1")
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--warmup_pct", type=float, default=0.2)
    parser.add_argument("--decay_type", type=str, default="trapezoidal")
    parser.add_argument("--decay_pct", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--total_samples", type=int, default=1_000_000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mlm_probability", type=float, default=0.3)
    parser.add_argument("--grad_accum", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()