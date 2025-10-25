#!/bin/bash
# Train Embeddageddon S Model (896D embeddings)
# Using tanh_l2_onecycle embeddings (30 epochs, lr 0.00001, weight decay 0.000005)

set -e  # Exit on any error

echo "============================================"
echo "Embeddageddon S Model Training (896D)"
echo "============================================"

# Create output directory
mkdir -p training_runs

# Train with embeddageddon embeddings
echo ""
echo "=== Training Embeddageddon S Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_s_896d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/embeddageddon/8bitopt_tanh_l2_onecycle_lower_ae_e30lr_embeddageddon_s_$(date +%Y%m%d_%H%M%S) \
    --batch_size 32 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --embedding_file data/embeddageddon_embeddings/tanh_l2_onecycle_e30_bs8192_lr0.00001_w8/embeddageddon_embeddings_s_896d.pkl \
    --random_subnetwork_order

echo ""
echo "============================================"
echo "Embeddageddon S training completed!"
echo "============================================"

exit

echo ""
echo "=== Training Equivalent S Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_s_896d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/plain/plain_8bitopt_matformer_s_$(date +%Y%m%d_%H%M%S) \
    --batch_size 32 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --random_subnetwork_order

echo ""
echo "============================================"
echo " Vanilla Matformer S training completed!"
echo "============================================"