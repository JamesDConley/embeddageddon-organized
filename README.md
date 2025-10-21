# Embedageddon
Good artists copy, great artists steal!

## Pull Embeddings
`python src/extract_embeddings.py`

## Setup Dataset
`python src/preprocess_embeddings.py --embedding_dir data/embedding_extraction/embedding_dicts --output_dir data/autoencoder/datasets/full`

## Train Autoencoder

Test with:
`python src/ae_train.py --embedding_dir data/embedding_extraction/small_test_dicts --epochs 3 --batch_size 256 --device cuda --output_dir data/autoencoder/trained_models/test_run_$(date +%Y%m%d_%H%M%S)`

Train for 100 Epochs with
`python src/ae_train.py --embedding_dir data/embedding_extraction/embedding_dicts --epochs 100 --batch_size 256 --device cuda --output_dir data/autoencoder/trained_models/epochs_100_$(date +%Y%m%d_%H%M%S)`

Train with dataset via
`python src/ae_train_memmap.py --preprocessed_dir data/autoencoder/datasets/full --batch_size 512 --num_workers 2 --epochs 100 --device cuda --output_dir data/autoencoder/trained_models/epochs_100_$(date +%Y%m%d_%H%M%S)`

# Generate Embeddageddon Embeddings
`python src/generate_embeddings.py --embedding_dicts_dir data/embedding_extraction/embedding_dicts --encoder_model_path data/autoencoder/trained_models/epochs_100_20251019_160346/models/embeddageddon_model_final.pth --output_dir data/embeddageddon_embeddings/xl --bottleneck_dim 7168`


# Setup a Training Dataset
`python src/generate_dataset.py --output_dir data/llm_datasets/redpajama_small`