#!/bin/bash

# Download Numerai dataset if not already present
python download.py --live
python download.py --train
python download.py --meta
python download.py --features

# Extract and Transform to sharded TFRecord files in ./data
python etl.py --train --split_valid --all --split 0.20 --shard 4096

# Create a baseline model for training
python naiMLP.py
python naiCNN.py

# Train the model(s)
bash train.sh

