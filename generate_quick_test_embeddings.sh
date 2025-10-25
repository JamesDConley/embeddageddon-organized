#!/bin/bash
# Generate embeddings from the tanh_l2_onecycle model (30 epochs, batch size 8192, lr 0.00001, L2 weight 0.01, weight decay 0.01, 8 workers)
# Model uses tanh bottleneck with L2 regularization, weight decay, and OneCycleLR scheduler

python src/generate_embeddings.py \
  --embedding_dicts_dir data/embedding_extraction/embedding_dicts \
  --encoder_model_path data/autoencoder/trained_models/tanh_l2_onecycle_e30_bs8192_lr0.00001_w8_20251023_101939/models/embeddageddon_model_final.pth \
  --output_dir data/embeddageddon_embeddings/tanh_l2_onecycle_e30_bs8192_lr0.00001_w8 \
  --bottleneck_dim 7168 \
  --use_tanh