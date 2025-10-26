# Build the image
docker build -t embeddageddon .

# Run with GPU support and the project mounted
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/app -v /big_data/hface_home:/root/.cache/huggingface embeddageddon

