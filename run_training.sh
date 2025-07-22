#!/bin/bash

caffeinate -is python src/train_modernbert_pt.py \
  --output_dir modernbert-pt \
  --checkpoint_dir checkpoints \
  --mlm_probability 0.3 \
  --max_length 2048 \
  --grad_accum 64 \
  --log_every 200 \
  --save_every 250_000 \
  --max_steps 7_500_001 \
  --warmup_steps 20_000 \
  --batch_size 4