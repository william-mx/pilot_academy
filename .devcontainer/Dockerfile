# Use the specific GPU + Jupyter image you requested
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Fix for potential GPG key issues and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update pip and reinstall TensorFlow with proper CUDA support
RUN pip install --upgrade pip
RUN pip uninstall -y tensorflow || true
RUN pip install tensorflow[and-cuda]

# Fix protobuf version warnings
RUN pip install --upgrade protobuf

# Install common DS libraries
RUN pip install numpy pandas matplotlib scikit-learn tqdm opencv-python

# Set the working directory
WORKDIR /workspaces/pilot_academy