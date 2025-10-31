#FROM nvcr.io/nvidia/pytorch:25.09-py3
#FROM nvcr.io/nvidia/pytorch:25.08-py3
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

RUN apt update && apt install git --yes

# Install cuDNN development headers required for TransformerEngine
RUN apt-get update && apt-get install -y libcudnn9-dev-cuda-12 && rm -rf /var/lib/apt/lists/*

# Install build dependencies for TransformerEngine
RUN pip3 install pybind11

# Install TransformerEngine from source with SM120 architecture support
RUN git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=120 pip3 install --no-build-isolation .

# Set the default command to bash
CMD ["/bin/bash"]