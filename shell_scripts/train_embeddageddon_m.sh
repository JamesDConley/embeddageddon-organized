#!/bin/bash
# Train Embeddageddon M Model (1792D embeddings)

set -e  # Exit on any error

echo "============================================"
echo "Embeddageddon M Model Training (1792D)"
echo "============================================"

# Create output directory
mkdir -p training_runs

# Train with embeddageddon embeddings
echo ""
echo "=== Training Embeddageddon M Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_m_1792d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/embeddageddon/tanh_embeddageddon_m_$(date +%Y%m%d_%H%M%S) \
    --batch_size 4 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --embedding_file data/embeddageddon_embeddings/100_epochs_8192_batch/embeddageddon_embeddings_m_1792d.pkl \
    --random_subnetwork_order

echo ""
echo "============================================"
echo "Embeddageddon M training completed!"
echo "============================================"


echo ""
echo "=== Training Equivalent M Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_m_1792d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/plain/plain_matformer_m_$(date +%Y%m%d_%H%M%S) \
    --batch_size 4 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --random_subnetwork_order

echo ""
echo "============================================"
echo " Vanilla Matformer M training completed!"
echo "============================================"
