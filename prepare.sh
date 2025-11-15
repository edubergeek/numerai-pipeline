#!/bin/bash

# To force new downloads set the environment variable FORCE to --force
# Download Numerai dataset if not already present
#python download.py --live ${FORCE}
python download.py --train ${FORCE}
python download.py --meta ${FORCE}
python download.py --features ${FORCE}

# Extract and Transform to sharded TFRecord files in ./data
python etl.py --train --split_valid --all --split 0.20 --shard 4096
#python etl.py --train --split_valid --all --split 0.02 --shard 4096 --features features.json --dir ./data/balance --balance


# Create a baseline model for training
python naiMLP.py
python naiCNN.py

# Train the model(s)
bash train.sh

