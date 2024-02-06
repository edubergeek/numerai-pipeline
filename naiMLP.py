"""
Implement a neural network model for Numerai data using a keras multi-layer perceptron.
To get started, install the required packages: pip install pandas numpy tensorflow keras
"""
# !pip install fastparquet numerapi

# +
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from naikeras import NumeraiKeras
from naiAPI import NumeraiAPI
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical

# +
""" 
Example of an artificial neural network model
A fully connected neural network or "multilayer percpetron" (MLP)
Derived from NumeraiKeras class
"""
class NumeraiMLP(NumeraiKeras):
  def __init__(self, hparam):
    super().__init__()

    # Construct the model
    inputShape = (2376)
    inputs = Input(shape=inputShape)
    self.BuildModel(inputs, hparam)

    self.SetLearningRate(hparam['learning_rate'])
    self.SetLoss('mae')
    self.SetMetrics(['mae','mse'])
    self.SetOptimizer(Adam)
    self.Compile()
  
    self.SetModelName("mlp")
    self.SetModelVersion(1)
    self.SetModelTrial(1)
    
    
  def MLP(self, inputs, hp):
    # Construct an MLP model

    m = inputs
    
    self.dropout=np.full((hp['layers']),hp['dropout'])
    batchNorm = hp['batch_norm']
    act = hp['activation']
    units = hp['units']
    layers = hp['layers']
    expansion = hp['expansion']
    for layer in range(layers):
      m = Dense(units)(m)
      if batchNorm:
        m = BatchNormalization()(m)
      m = activations.get(act)(m)
      if not batchNorm and self.dropout[layer] > 0.0:
        m = Dropout(self.dropout[layer])(m)
      units *= expansion

    m = Dense(hp['final_units'])(m)
    m = activations.get(hp['final_activation'])(m)
    if batchNorm:
      m = BatchNormalization()(m)
    if not batchNorm and hp['final_dropout'] > 0.0:
      m = Dropout(hp['final_dropout'])(m)

    m = Dense(5, activation=hp['final_activation'])(m)
    target = Dense(1, activation=hp['target_activation'], name='target')(m)
    
    return target
    
  def BuildModel(self, inputs, hp):
    # Construct the model

    target = self.MLP(inputs, hp)
    m = Model(inputs=inputs, outputs=[target], name="mlp")
    self.SetModel(m)
    return m
    
def main():
  hyperParam = {
    'oldmodel': False,
    'units': 1024,
    'expansion': 0.5,
    'layers': 6,
    'activation': 'relu',
    'batch_norm': True,
    'dropout': 0.0,
    'final_units': 32,
    'final_dropout': 0.0,
    'final_activation': 'relu',
    'target_activation': 'relu',
    'learning_rate': 0.001,
  }

  napi = NumeraiAPI()
  nm = NumeraiMLP(hyperParam)
  print(nm.model.summary())

  nm.SetModelTrial(1)
  nm.SetModelRevision(2)
  nm.SetModelPath("./"+nm.GetModelName())
  nm.SaveModel()


# -
if __name__ == '__main__':
  main()


