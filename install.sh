#!/bin/bash

set -e

echo "Starting Petals installation..."

# Check if running on Debian/Ubuntu
if [ -f /etc/debian_version ]; then
    echo "Debian/Ubuntu detected. Installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip git pciutils
else
    echo "Warning: Not running on Debian/Ubuntu. Please ensure python3, python3-venv, python3-pip, git, and pciutils are installed."
fi

# Create virtual environment
ENV_DIR="petals-env"
echo "Creating virtual environment in $ENV_DIR..."
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Detect GPU
GPU_TYPE="CPU"
echo "Detecting GPU..."
if command -v lspci &> /dev/null; then
    if lspci | grep -i "nvidia" | grep -i "vga\|3d\|display" &> /dev/null; then
        GPU_TYPE="NVIDIA"
    elif lspci | grep -i "amd" | grep -i "vga\|3d\|display" &> /dev/null; then
        GPU_TYPE="AMD"
    fi
else
    if command -v nvidia-smi &> /dev/null; then
        GPU_TYPE="NVIDIA"
    elif command -v rocm-smi &> /dev/null; then
        GPU_TYPE="AMD"
    fi
fi

echo "Detected GPU Type: $GPU_TYPE"

# Install PyTorch
echo "Installing PyTorch..."
if [ "$GPU_TYPE" = "NVIDIA" ]; then
    pip install torch
elif [ "$GPU_TYPE" = "AMD" ]; then
    pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# Install Petals
echo "Installing Petals..."
pip install .

echo ""
echo "========================================="
echo "        Installation Complete!           "
echo "========================================="
echo ""
echo "You're all set! To activate your environment and start a server, run:"
echo ""
echo "    source $ENV_DIR/bin/activate"
echo "    python -m petals.cli.run_server meta-llama/Meta-Llama-3.1-405B-Instruct"
echo ""
echo "========================================="
