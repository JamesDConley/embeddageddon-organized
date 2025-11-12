#!/bin/bash
# Production run script: Downloads datasets from HuggingFace using fast transfer

set -e  # Exit on error

echo "Starting dataset downloads from HuggingFace with fast transfer enabled..."

# Enable HF_TRANSFER for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create data directories if they don't exist
mkdir -p data/fineweb-s
mkdir -p data/fineweb-m

# Download fineweb-sample-5.97B-512 to data/fineweb-s
echo "Downloading JamesConley/fineweb-sample-5.97B-512 to data/fineweb-s..."
huggingface-cli download \
  --repo-type dataset \
  --local-dir data/fineweb-s \
  JamesConley/fineweb-sample-5.97B-512

echo "✓ Downloaded fineweb-sample-5.97B-512"

# Download fineweb-sample-22.95B-512 to data/fineweb-m
echo "Downloading JamesConley/fineweb-sample-22.95B-512 to data/fineweb-m..."
huggingface-cli download \
  --repo-type dataset \
  --local-dir data/fineweb-m \
  JamesConley/fineweb-sample-22.95B-512

echo "✓ Downloaded fineweb-sample-22.95B-512"

# Set open permissions for the data folder
echo "Setting permissions on data folder..."
chmod -R 777 data

echo "All datasets downloaded successfully!"