from datasets import load_dataset
from itertools import islice
from torch.utils.data import DataLoader, IterableDataset
from transformers import DataCollatorForLanguageModeling
import logging

class TokenizedIterableDataset(IterableDataset):
    def __init__(self, data_iter):
        self.data_iter = data_iter
    def __iter__(self):
        return iter(self.data_iter)


def stream_tokenizer(tokenizer, data_stream, start_step, max_length):
    for i, ex in enumerate(data_stream):
        yield tokenizer(
            ex['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )


def get_dataloader(
    args, tokenizer, start_step: int
) -> DataLoader:
    ds = load_dataset(
        args.dataset_name, split='train', streaming=True
    )
    part = [0, 30_000_000] if args.train_part == 'pt1' else [30_000_000, None]
    total_samples = part[1]

    logging.info(f"Getting data from {part[0]}th sentence to {part[1]}th")

    sliced = islice(ds, part[0], part[1])
    tokenized = stream_tokenizer(
        tokenizer, sliced, start_step, args.max_length
    )
    iterable = TokenizedIterableDataset(tokenized)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        mask_replace_prob=args.mask_replace_prob,
        random_replace_prob=args.random_replace_prob
    )
    return (
        DataLoader(
        iterable,
        batch_size=args.batch_size,
        collate_fn=collator
        ),
        total_samples
    )