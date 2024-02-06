#!/bin/bash

# Download Numerai dataset if not already present
python download.py --live
python download.py --train
python download.py --meta
python download.py --features

# Extract and Transform to sharded TFRecord files in ./data
python etl.py --train --split 0.04 --shard 1024

# Create a baseline model for training
python naiMLP.py

# Train the model(s)
bash train.sh

