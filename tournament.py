"""
Get the current round and print it out.
"""
# !pip install fastparquet numerapi

# +
import os
from pathlib import Path
from naiAPI import NumeraiAPI

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

# -
if __name__ == '__main__':
  main()


