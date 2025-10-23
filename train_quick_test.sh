#!/bin/bash
# Quick test training run: 20 epochs, batch size 8192, lr 0.0005, 8 workers, L2 penalty 0.01, weight decay 0.01
# Using tanh bottleneck with L2 regularization on embeddings and OneCycleLR scheduler

python src/ae_train_memmap.py \
  --preprocessed_dir /fast/embeddageddon_datasets/full/ \
  --epochs 30 \
  --batch_size 8192 \
  --learning_rate 0.00001 \
  --num_workers 8 \
  --l2_penalty_weight 0.0 \
  --weight_decay 0.000005 \
  --use_tanh \
  --device cuda \
  --output_dir data/autoencoder/trained_models/tanh_l2_onecycle_e30_bs8192_lr0.00001_w8_$(date +%Y%m%d_%H%M%S)