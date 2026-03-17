#!/usr/bin/env bash
set -e

# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade build tooling
python -m pip install --upgrade pip setuptools wheel

# Install Apple Silicon TensorFlow first
pip install tensorflow-macos tensorflow-metal

# Install remaining dependencies
pip install -r requirements.txt

python -m ipykernel install --user --name=recommender_env --display-name "Python (recommender)"

echo "Setup complete. Activate with: source .venv/bin/activate"
