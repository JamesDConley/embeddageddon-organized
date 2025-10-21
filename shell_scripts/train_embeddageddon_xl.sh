#!/bin/bash
# Train Embeddageddon XL Model (7168D embeddings)

set -e  # Exit on any error

echo "============================================"
echo "Embeddageddon XL Model Training (7168D)"
echo "============================================"

# Create output directory
mkdir -p training_runs

# Train with embeddageddon embeddings
echo ""
echo "=== Training Embeddageddon XL Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_xl_7168d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/embeddageddon/tanh_embeddageddon_xl_$(date +%Y%m%d_%H%M%S) \
    --batch_size 2 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --embedding_file data/embeddageddon_embeddings/100_epochs_8192_batch/embeddageddon_embeddings_xl_7168d.pkl \
    --use_fp8 \
    --random_subnetwork_order

echo ""
echo "============================================"
echo "Embeddageddon XL training completed!"
echo "============================================"


echo ""
echo "=== Training Equivalent XL Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_xl_7168d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/plain/plain_matformer_xl_$(date +%Y%m%d_%H%M%S) \
    --batch_size 2 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --use_fp8 \
    --random_subnetwork_order

echo ""
echo "============================================"
echo " Vanilla Matformer XL training completed!"
echo "============================================"
