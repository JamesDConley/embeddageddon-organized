# Build the image
docker build -t embeddageddon .

# Run with GPU support and the project mounted
docker run --gpus all -it -v $(pwd):/app embeddageddon

