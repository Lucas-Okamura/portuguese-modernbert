#!/bin/bash

accelerate launch --multi_gpu modernbert_trainer/train.py \
  --optimizer stableadamw \
  --train_part pt1 \
  --checkpoint_dir checkpoints \
  --rope_theta 10_000.0 \
  --total_samples 34_841_241 \
  --epochs 2 \
  --mlm_probability 0.3 \
  --max_length 1024 \
  --grad_accum 72 \
  --log_every 200 \
  --save_every 50_000 \
  --warmup_pct 0.05 \
  --decay_pct 0.3 \
  --lr 8e-4 \
  --min_lr 1e-8 \
  --batch_size 16