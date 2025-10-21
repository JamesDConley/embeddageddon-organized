#!/bin/bash
# Train Embeddageddon L Model (3584D embeddings)

set -e  # Exit on any error

echo "============================================"
echo "Embeddageddon L Model Training (3584D)"
echo "============================================"

# Create output directory
mkdir -p training_runs

# Train with embeddageddon embeddings
echo ""
echo "=== Training Embeddageddon L Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_l_3584d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/embeddageddon/tanh_embeddageddon_l_$(date +%Y%m%d_%H%M%S) \
    --batch_size 1 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --use_fp8 \
    --embedding_file data/embeddageddon_embeddings/100_epochs_8192_batch/embeddageddon_embeddings_l_3584d.pkl \
    --random_subnetwork_order

echo ""
echo "============================================"
echo "Embeddageddon L training completed!"
echo "============================================"


echo ""
echo "=== Training Equivalent L Model ==="
python src/llm_train.py \
    --model_type matformer \
    --config_name llm_configs/embeddageddon_l_3584d.json \
    --dataset_dir ../MatFormer/matformer/datasets/red_pajama \
    --output_dir data/language_models/plain/plain_matformer_l_$(date +%Y%m%d_%H%M%S) \
    --batch_size 1 \
    --max_length 512 \
    --num_epochs_per_subnetwork 0.25 \
    --learning_rate 1e-4 \
    --seed 42 \
    --use_fp8 \
    --random_subnetwork_order

echo ""
echo "============================================"
echo " Vanilla Matformer L training completed!"
echo "============================================"
