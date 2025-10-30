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

# Install TransformerEngine
#RUN NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
RUN NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.9

# Set the default command to bash
CMD ["/bin/bash"]