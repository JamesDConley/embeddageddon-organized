FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV PATH=/usr/local/cuda-13.0/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda-13.0

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-build-isolation transformer_engine[pytorch]
RUN pip install --no-cache-dir -r requirements.txt
# RUN git clone --branch main --recursive https://github.com/NVIDIA/TransformerEngine.git && \
#     cd TransformerEngine && \
#     CMAKE_CUDA_ARCHITECTURES="80;90;100;120" \
#     NVTE_FRAMEWORK=pytorch \
#     python -m pip install --no-build-isolation .

# Set the default command to bash
CMD ["/bin/bash"]