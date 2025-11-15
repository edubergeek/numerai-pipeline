"""
Implement a neural network model for Numerai data using a keras multi-layer perceptron.
To get started, install the required packages: pip install pandas numpy tensorflow keras
"""
# !pip install fastparquet numerapi

import os
import argparse
from pathlib import Path
from naiAPI import NumeraiAPI

# +
parser = argparse.ArgumentParser()
parser.add_argument("--version", default='5.0', help="data release version")
parser.add_argument("--train", action='store_true', help="download training set")
parser.add_argument("--valid", action='store_true', help="download validation data")
parser.add_argument("--live", action='store_true', help="download live data")
parser.add_argument("--meta", action='store_true', help="download metamodel")
parser.add_argument("--force", action='store_true', help="overwrite existing file(s)")
parser.add_argument("--features", action='store_true', help="download features")

args = parser.parse_args()
print(args)

# +
""" 
Run tournament pipeline
"""

class NumeraiRound(NumeraiAPI):
  def __init__(self):
    super().__init__()

# --


def main():
  r = NumeraiRound()
  print(r.GetRound())
  #r.ListData()
  r.SetVersion(args.version)    
  if args.train:
    if args.force or not os.path.exists(r.TRAINDATA): 
      r.DownloadTrain(verbose=True)
    if args.force or not os.path.exists(r.VALIDDATA): 
      r.DownloadValid(verbose=True)
  if args.valid:
    r.DownloadValid(verbose=True)
  if args.live:
    r.DownloadLive(verbose=True)
  if args.meta:
    if args.force or not os.path.exists(r.METAMODEL): 
      r.DownloadMetaModel(verbose=True)
  if args.features:
    if args.force or not os.path.exists(r.FEATURES): 
      r.DownloadFeatures(verbose=True)

# -
if __name__ == '__main__':
  main()


