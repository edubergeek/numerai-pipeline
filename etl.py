# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# #!pip install pyarrow
# #!pip install sklearn scikit-image
# -

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import json
import argparse
#import csv
import tensorflow as tf
import skimage.io as skio
from sklearn.model_selection import train_test_split
from enum import IntEnum
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from naiETL import _floatvector_feature, _float_feature, _int64_feature, _bytes_feature, _dtype_feature, TrainingSet, ETL, NumeraiETL

# +
IS_JUPYTER = False
#--train --split_valid --all --split 0.20 --shard 4096

if IS_JUPYTER:
  sys.argv.append('--train')
  sys.argv.append('--split_valid')
  sys.argv.append('--all')
  sys.argv.append('--split')
  sys.argv.append('0.05')
  sys.argv.append('--shard')
  sys.argv.append('4096')
  #sys.argv.append('--eras')
  #sys.argv.append('--balance')
  #sys.argv.append('--split')
  #sys.argv.append('0.05')
  #sys.argv.append('--shard')
  #sys.argv.append('2048')



# +


parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="ETL training set")
parser.add_argument("--all", action='store_true', help="Combine training and validation examples")
parser.add_argument("--live", action='store_true', help="ETL live data")
parser.add_argument("--eras", action='store_true', help="ETL by era")
parser.add_argument("--split_valid", action='store_true', help="Apply split to validation set only")
parser.add_argument("--balance", action='store_true', help="balance target values by era")
parser.add_argument("--shard", type=int, default=2048, help="shard size")
parser.add_argument("--split", type=float, default=0.1, help="validation split percentage")
parser.add_argument("--dir", default='./data', help="directory to save TFRecord files in")




# +
print(sys.argv)
if IS_JUPYTER:
  args = parser.parse_args(sys.argv[3:])
else:
  args = parser.parse_args()

print(args)

# -

dataRoot = args.dir
if not os.path.exists(dataRoot):
  os.makedirs(dataRoot)

# +
# #!rm -f data/*
# -

etl = NumeraiETL('.', dataRoot, valid_split = args.split, shard_size = args.shard)

# If in balance mode don't step on train, valid, test data file prefixes
if args.balance:
  etl.outputPart = [None, "balance", "balanceval", "ignore"]

etl.OpenDatasets(byEra=args.eras, balance=args.balance)

if args.train:
  # Load training examples with train/valid split
  print("Loading training dataset ...")
  reload = not args.all
  etl.Load(TrainingSet.TRAIN, reload=reload)
  print("Sharding %d training examples ..." % (etl.examples[TrainingSet.TRAIN]))
  etl.SaveDataset(TrainingSet.TRAIN, TrainingSet.TRAIN, byEra=args.eras, balance=args.balance)
  if args.all:
    print("Sharding %d validation as training examples ..." % (etl.examples[TrainingSet.VALID]))
    etl.SaveDataset(TrainingSet.TRAIN, TrainingSet.VALID, byEra=args.eras, balance=args.balance)
  # Load validation examples with train/valid split
  print("Loading validation dataset ...")
  etl.Load(TrainingSet.VALID, reload=reload)
  if args.all:
    print("Sharding %d training examples ..." % (etl.examples[TrainingSet.TRAIN]))
    etl.SaveDataset(TrainingSet.VALID, TrainingSet.TRAIN, byEra=args.eras, balance=args.balance)

  print("Sharding %d validation examples ..." % (etl.examples[TrainingSet.VALID]))
  etl.SaveDataset(TrainingSet.VALID, TrainingSet.VALID, byEra=args.eras, balance=args.balance)

if args.live:
  etl.Load(TrainingSet.TEST)
  etl.SaveDataset(TrainingSet.TEST, TrainingSet.TEST, args.eras)

etl.CloseDatasets()
print("Finished")

