#!/bin/bash
set -e  # Exit immediately if any command fails

# Build the image
docker build -t embeddageddon .

# Run with GPU support and the project mounted
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/app embeddageddon

