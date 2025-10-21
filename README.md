# Embedageddon
Good artists copy, great artists steal!

## Pull Embeddings
`python src/extract_embeddings.py`

## Setup Dataset
`python src/preprocess_embeddings.py --embedding_dir data/embedding_extraction/embedding_dicts --output_dir data/autoencoder/datasets/full`

## Train Autoencoder


Train with dataset via
`python src/ae_train_memmap.py --preprocessed_dir data/autoencoder/datasets/full --batch_size 512 --num_workers 2 --epochs 100 --device cuda --output_dir data/autoencoder/trained_models/epochs_100_$(date +%Y%m%d_%H%M%S)`

You can also use the `ae_train_preloaded.py` if you have enough memory to fit all of the dictionaries into RAM. Note this will happen per worker, and multiple workers are needed to keep up with a single RTX Pro 6000. Even with it all preloaded into memory.

The memmap script I also find to get poor GPU utilization. There's definitely optimization that can be done here feeding the data to the training code.

# Generate Embeddageddon Embeddings
`python src/generate_embeddings.py --embedding_dicts_dir data/embedding_extraction/embedding_dicts --encoder_model_path data/autoencoder/trained_models/epochs_100_20251019_160346/models/embeddageddon_model_final.pth --output_dir data/embeddageddon_embeddings/xl --bottleneck_dim 7168`

`python src/generate_embeddings.py --embedding_dicts_dir data/embedding_extraction/embedding_dicts --encoder_model_path data/autoencoder/trained_models/no_dropout_epochs_100_20251021_092845/models/embeddageddon_model_final.pth --output_dir data/embeddageddon_embeddings/no_dropout --bottleneck_dim 7168`

`python src/generate_embeddings.py --embedding_dicts_dir data/embedding_extraction/embedding_dicts --encoder_model_path data/autoencoder/trained_models/no_dropout_epochs_100_20251021_092845/checkpoints/embeddageddon_model_epoch_50.pth --output_dir data/embeddageddon_embeddings/50_epochs_no_dropout --bottleneck_dim 7168`



# Setup a Training Dataset
`python src/generate_dataset.py --output_dir data/llm_datasets/redpajama_small`