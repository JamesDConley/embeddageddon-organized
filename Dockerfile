FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to bash
CMD ["/bin/bash"]