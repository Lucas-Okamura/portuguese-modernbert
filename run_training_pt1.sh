#!/bin/bash

accelerate launch --multi_gpu modernbert_trainer/train.py \
  --output_dir modernbert-pt \
  --train_part pt1 \
  --checkpoint_dir checkpoints \
  --rope_theta 10_000.0 \
  --mlm_probability 0.3 \
  --max_length 1024 \
  --grad_accum 64 \
  --log_every 200 \
  --save_every 200_00 \
  --max_steps 2_220_000 \
  --warmup_steps 300 \
  --batch_size 16