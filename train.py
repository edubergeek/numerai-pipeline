#!/usr/bin/env python
# coding: utf-8
# %%
import sys
import os
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, AdamW
from keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, Concatenate, BatchNormalization, Conv1DTranspose
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, Conv3D
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# %%
from numerapi import NumerAPI
from naiAPI import NumeraiAPI
from naikeras import NumeraiKeras, PlotLoss, BestEpoch
from naiETL import _floatvector_feature, _float_feature, _int64_feature, _bytes_feature, _dtype_feature, TrainingSet, ETL, NumeraiETL

# %%
IS_JUPYTER = False
TRANSFORM = 'transform'
#Set to index position of the desired training target
TARGET_IDX = 0

def decode_tfr(record_bytes):
    schema =  {
      "era": tf.io.FixedLenFeature([], dtype=tf.int64),
      "id": tf.io.FixedLenFeature([], dtype=tf.string),
      "x":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
      "y":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
    }    
    example = tf.io.parse_single_example(record_bytes, schema)
    return example['x'], example['y'][TARGET_IDX]
 
def reshape_C6(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'C6':
        # clone the last 50 values 4 times 50 50 50 50
        temp1 = tf.slice(features, [0], [1000])
        temp2 = tf.slice(features, [1000], [50])
        temp3 = tf.repeat(temp2, 4)
        temp4 = tf.concat([temp1, temp3], 0)
        features = temp4
        #features = tf.concat((tf.slice(features, [0,1000]), tf.repeat(tf.slice(features, [1000, 1050]), 4)), 1)
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features = tf.reshape(features, [6, yDim, xDim])
        features = tf.transpose(features, perm=[1, 2, 0])
    return features, targets

def reshape_ZYX(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'ZYX':
        zDim = int(t['arg1'])
        yDim = int(t['arg2'])
        xDim = int(t['arg3'])        
        features = tf.reshape(features, [zDim, yDim, xDim])
        #features = tf.transpose(features, perm=[1, 2, 0])
        #features = tf.transpose(features)
    return features, targets

def reshape_YXZ(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'YXZ':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        zDim = int(t['arg3'])        
        features = tf.reshape(features, [yDim, xDim, zDim])
        #features = tf.transpose(features)
    return features, targets

def reshape_XY(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'XY':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features = tf.reshape(features, [xDim, yDim])
        features = tf.transpose(features)
    return features, targets

def reshape_YX(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'YX':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features = tf.reshape(features, [yDim, xDim])
    return features, targets

def reshape_Slice(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Slice':
        fromDim = int(t['arg1'])
        toDim = int(t['arg2'])
        features = tf.slice(features, [fromDim], [toDim])
    return features, targets

def reshape_Even(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Even':
        fromDim = int(t['arg1'])
        toDim = int(t['arg2'])
        features = tf.gather(features, tf.constant(np.arange(fromDim, toDim*2, 2)))
        features = tf.slice(features, [fromDim], [toDim])
    return features, targets

def reshape_Pad(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Pad':
        fromPos = int(t['arg1'])
        toPos = int(t['arg2'])
        featureVal = float(t['arg3'])
        # initialize an array of "featureVal" values in the correct shape
        #val = np.ones((toPos-fromPos)) * featureVal
        #pad = tf.constant(val, dtype=tf.float32)
        pad = tf.slice(features, [0], [toPos-fromPos])
        features = tf.concat([features, pad], 0)
        
    return features, targets

def replace_NaN(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'NaN':
        featureVal = float(t['arg1'])
        targetVal = float(t['arg2'])
        mask = tf.math.is_nan(features)
        maskVal = tf.cast(tf.ones_like(mask), tf.float32) * tf.constant(featureVal, dtype=tf.float32)
        features = tf.where(mask, maskVal, features)
        mask = tf.math.is_nan(targets)
        maskVal = tf.cast(tf.ones_like(mask), tf.float32) * tf.constant(targetVal, dtype=tf.float32)
        targets = tf.where(mask, maskVal, targets)
    return features, targets

def replace_Sparse(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Sparse':
        nClass = float(t['arg1'])
        #targetVal = float(t['arg2'])
        #targets = targets * tf.constant(nClass, dtype=tf.float32)
        targets = tf.one_hot(targets, nClass)
    return features, targets

def remap_autoencoder(features, targets):
    return features, (features, targets)
 


# %%
class WarmUpCosine(LearningRateSchedule):
  def __init__(
    self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
  ):
    super().__init__()

    self.learning_rate_base = learning_rate_base
    self.total_steps = total_steps
    self.warmup_learning_rate = warmup_learning_rate
    self.warmup_steps = warmup_steps
    self.pi = tf.constant(np.pi)

  def __call__(self, step):
    if self.total_steps < self.warmup_steps:
      raise ValueError("Total_steps must be larger or equal to warmup_steps.")

    cos_annealed_lr = tf.cos(self.pi * (tf.cast(step, tf.float32) - self.warmup_steps) / float(self.total_steps - self.warmup_steps))
    learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

    if self.warmup_steps > 0:
      if self.learning_rate_base < self.warmup_learning_rate:
        raise ValueError(
          "Learning_rate_base must be larger or equal to "
          "warmup_learning_rate."
          )
      slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
      warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
      learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)
    return tf.where(step > self.total_steps, 0.0, learning_rate, name="learning_rate")


# %%
class NumeraiModel():
  def __init__(self, modelname, hyperParam):
    self.sModelPath = "../round/%d" %(hyperParam["round"])
    self.sModelName = modelname
    self.modelStep = 0
    self.modelVersion = hyperParam["version"]
    self.modelRevision = hyperParam["revision"]
    self.modelTrial = hyperParam["trial"]
    self.modelEpoch = hyperParam["epoch"]
    self.sModelSuffix = ""
    self.earlyStopPatience = hyperParam["patience"]
    self.earlyStopThreshold = hyperParam["threshold"]
    self.batchSize = hyperParam["batch_size"]
    self.targetX = [0.1, 0.48, 0.0, 0.0, 0.0]
    self.dft = None
    self.epochs = hyperParam["epochs"]
    self.begin = hyperParam["begin"]
    self.monitor = 'val_loss'
    self.loss = 'mse'
    self.mode = 'min'
    self.isTrained = False
    self.useTensorboard = False

    self.hparam = hyperParam

  def LoadModel(self):
    self.modelFile = self.GetModelFile()
    print("Loading ", self.modelFile)
    if self.hparam['arch'] == 'XGB':
      self.model = XGBRegressor(max_depth=8, learning_rate=7e-3, n_estimators=6000, n_jobs=18, colsample_bytree=0.1)
      self.model.load_model(self.modelFile)
    elif self.hparam['arch'] == 'LGB':
      self.model = lgb.Booster(model_file=self.modelFile)
    else:
      self.model = tf.keras.models.load_model(self.modelFile)
      if self.hparam['arch'] == 'NC' or self.hparam['arch'] == 'CC':
        self.Compile(self.model, loss='sparse_categorical_crossentropy', metric=['accuracy'])
      else:
        self.Compile(self.model)

  def GetModelFile(self):
    return "%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, self.sModelName, self.modelVersion, self.modelRevision, self.modelTrial, self.modelEpoch, self.sModelSuffix)

  def GetBestEpochFile(self):
    return "%s/%s-r%d-best.csv" %(self.sModelPath, self.sModelName, self.modelRevision)

  def LoadBestEpochs(self, predict='-'):
    self.bestEpochFile = self.GetBestEpochFile()
    print('BestEpochFile = ', self.bestEpochFile)
    if os.path.exists(self.bestEpochFile):
      self.best = np.loadtxt(self.bestEpochFile, delimiter=',', dtype='int')
    else:
      self.bestEpochFile = "%s/%s-r%d-best.csv" %(self.sModelPath, predict, self.modelRevision)
      if os.path.exists(self.bestEpochFile):
        self.best = np.loadtxt(self.bestEpochFile, delimiter=',', dtype='int')
      else:
        self.best = np.ones((41))
    print(self.best)

  def GetTrialModelFile(self, model, version, trial, epoch, predict):
    if predict == '-':
      return "%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, model, version, self.modelRevision, trial, epoch, self.sModelSuffix)
    else:
      return "%s/%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, predict, model, version, self.modelRevision, trial, epoch, self.sModelSuffix)

  def GetModelFullName(self):
    if self.modelStep:
      return "%sv%dr%ds%d" %(self.sModelName, self.modelVersion, self.modelRevision, self.modelStep)
    return "%sv%dr%d" %(self.sModelName, self.modelVersion, self.modelRevision)

  def GetModelName(self):
    return self.sModelName

  def GetModelTrial(self):
    return self.modelTrial

  def SetModelTrial(self, trial):
    self.modelTrial = trial

  def GetModelStep(self):
    return self.modelStep

  def SetModelStep(self, step):
    self.modelStep = step

  def GetModelRevision(self):
    return self.modelRevision

  def GetList(self, f):
    return f.numpy().tolist()

  def SetTensorboard(self, useTensorboard = True):
    self.useTensorboard = useTensorboard

  def SetLoss(self, loss):
    self.loss = loss

  def SetMonitor(self, monitor):
    self.monitor = monitor
    if monitor == 'val_accuracy':
      self.mode = 'max'

  def GetDataSet(self, filenames, transform):
    at = AUTOTUNE
    
    dataset = (
      tf.data.TFRecordDataset(filenames, num_parallel_reads=at)
      .map(decode_tfr, num_parallel_calls=at)
    )
    
    if not transform == '-':
      for t in self.hparam[transform]:
        if t['name'] == 'C6':
          dataset = dataset.map(reshape_C6, num_parallel_calls=at)
        if t['name'] == 'XY':
          dataset = dataset.map(reshape_XY, num_parallel_calls=at)
        if t['name'] == 'YX':
          dataset = dataset.map(reshape_YX, num_parallel_calls=at)
        if t['name'] == 'YXZ':
          dataset = dataset.map(reshape_YXZ, num_parallel_calls=at)
        if t['name'] == 'ZYX':
          dataset = dataset.map(reshape_ZYX, num_parallel_calls=at)
        if t['name'] == 'Even':
          dataset = dataset.map(reshape_Even, num_parallel_calls=at)
        if t['name'] == 'Slice':
          dataset = dataset.map(reshape_Slice, num_parallel_calls=at)
        if t['name'] == 'NaN':
          dataset = dataset.map(replace_NaN, num_parallel_calls=at)
        if t['name'] == 'Pad':
          dataset = dataset.map(reshape_Pad, num_parallel_calls=at)
        if t['name'] == 'Sparse':
          dataset = dataset.map(replace_Sparse, num_parallel_calls=at)

    if self.hparam['arch'] == 'AE':
      dataset = dataset.map(remap_autoencoder, num_parallel_calls=at)
    
    dataset = dataset.batch(self.batchSize).prefetch(at).repeat(count=1)

    return dataset
       
  def DataSet(self, path, pattern):
    pattern_list = pattern.split()
    filenames = []
    for pat in pattern_list:
      filenames += tf.io.gfile.glob(os.path.join(path, pat))
    TRANSFORM='transform'
    return self.GetDataSet(filenames, TRANSFORM)

  def SaveSubmissionCSV(self):
    # Save predictions as a CSV and upload to https://numer.ai
    submissionCSV = self.GetModelFullName() + ".csv"
    if (self.hparam['arch'] == 'AE' or self.hparam['arch'] == 'TF') and self.dft['prediction'].ndim == 1:
      pred = [x for row in self.dft['prediction'] for x in row]
      self.dft['prediction'] = pred
    self.dft.to_csv(submissionCSV, header=True, index=False)
    return submissionCSV

  def ResetWeights(self):
    if self.isTrained:
      weights = []
      initializers = []
      for layer in self.model.layers:
        #if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
        if isinstance(layer, (Dense, Conv1D, Conv2D, Conv1DTranspose)):
          weights += [layer.kernel, layer.bias]
          initializers += [layer.kernel_initializer, layer.bias_initializer]
        elif isinstance(layer, BatchNormalization):
          weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
          initializers += [layer.gamma_initializer,
                       layer.beta_initializer,
                       layer.moving_mean_initializer,
                       layer.moving_variance_initializer]
      for w, init in zip(weights, initializers):
        w.assign(init(w.shape, dtype=w.dtype))

    self.isTrained = False

  def LRDecay(self, initial_lr = 0):
    lr_warmup_decayed_fn = CosineDecay(initial_lr, decay_steps=self.hparam['decay_steps'], warmup_target=self.hparam['lr'], warmup_steps=self.hparam['warmup'])
    return lr_warmup_decayed_fn

  def Compile(self, model, loss=None, metric=None, loss_weight=None):
    # compile the model  

    if self.hparam['arch'] == 'CC':
      total_steps = int(self.hparam['steps'] / self.batchSize * self.epochs)
      warmup_epoch_percentage = 0.10
      warmup_steps = int(total_steps * warmup_epoch_percentage)
      #scheduled_lrs = WarmUpCosine(learning_rate_base=self.learning_rate,total_steps=total_steps, warmup_learning_rate=0.0, warmup_steps=warmup_steps)
      #scheduled_lrs = LRDecay()
      self.optimizer = AdamW(learning_rate=self.hparam['lr'], weight_decay=self.hparam['decay'])
    else:
      self.optimizer = Adam(learning_rate=self.hparam['lr'])
    if self.hparam['arch'] == 'AE':
      losses = {
        "target_output": self.loss,
        "input_output": self.loss,
      }
      lossWeights = {"target_output": 1.0, "input_output": self.hparam['epsilon']}
      model.compile(optimizer=self.optimizer, loss=losses, loss_weights=lossWeights, metrics=metric)
    elif self.hparam['arch'] == 'CC':
      loss = SparseCategoricalCrossentropy(from_logits=True)
      metrics = [SparseCategoricalAccuracy(name="accuracy"),]
      model.compile( optimizer=self.optimizer, loss=loss, metrics=metrics)
    else:
      if loss is None:
        loss = self.loss
      if metric is None:
        metric = [loss]
      model.compile(optimizer=self.optimizer, loss=loss, metrics=metric, loss_weights=loss_weight)
    return model

  def Train(self, ds, dsv):
    # Set the model file name
    filepath="%s/%st%d-e{epoch:d}%s" %(self.sModelPath, self.GetModelFullName(), self.GetModelTrial(), self.sModelSuffix)
    # default checkpoint settings
    checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False, mode=self.mode)
    # plot loss after each epoch
    bestepoch = BestEpoch(metric=self.monitor, mode=self.mode)
    self.bestEpoch = 0

    self.callbacks = [checkpoint, bestepoch]

    if self.earlyStopPatience > 0:
      earlystop = EarlyStopping(monitor=self.monitor, mode=self.mode, patience=self.earlyStopPatience, min_delta=self.earlyStopThreshold)
      self.callbacks.append(earlystop)

    if self.useTensorboard:
      tensorboard_callback = TensorBoard(log_dir="./logs")
      self.callbacks.append(tensorboard_callback)
    else:
      plotloss = PlotLoss(metric=self.monitor)
      self.callbacks.append(plotloss)
       
    # TODO train lgbm and xgboost models
    #if arch == 'XGB':
    #  model = XGBRegressor(max_depth=8, learning_rate=7e-3, n_estimators=6000, n_jobs=18, colsample_bytree=0.1)
    # model.Train()
    # elif arch == 'LGB':
    #  model.Train()
    if not (self.hparam['arch'] == 'AE' or self.hparam['arch'] == 'NC' or self.hparam['arch'] == 'NR' or self.hparam['arch'] == 'CC'):
      print("Unsupported arch=%s in Train" %(self.hparam['arch']))
      return 0
    
    self.model.fit(ds, validation_data=dsv, initial_epoch=self.begin, epochs=self.epochs, batch_size=self.batchSize, callbacks=self.callbacks, verbose=1, shuffle=1)
    
    self.isTrained = True  
    self.bestEpoch = bestepoch.get_best_epoch()
    return self.bestEpoch

  def PredictTargetX(self, yPred):
        
    yhat = np.full((yPred.shape[0],1), 2)

    selected = yPred[:,2] <= self.targetX[0]
    a = yPred[:,0] >= yPred[:,4]
    b = yPred[:,4] > yPred[:,0]
    is0 = selected & a
    yhat[np.argwhere(is0)] = 0
    is4 = selected & b
    yhat[np.argwhere(is4)] = 4

    selected = yPred[:,2] <= self.targetX[1]
    a = yPred[:,1] >= yPred[:,3]
    b = yPred[:,3] > yPred[:,1]
    is1 = selected & a
    yhat[np.argwhere(is1)] = 1
    is3 = selected & b
    yhat[np.argwhere(is3)] = 3

    return yhat / 4.0

  def Predict(self, ds, transform):
    # Generate predictions
    b=0
    lid = []
    lpred = []
    names = ["id", "prediction"]

    for features in ds:
      yId = nm.GetList(features['id'])
      yId = list(yId)
      if self.hparam['arch'] == 'AE':
        _, yPred = nm.model.predict(features['x'], self.batchSize)
      else:
        if self.hparam['arch'] == 'XGB' or self.hparam['arch'] == 'LGB':
          yPred = self.model.predict(features['x'].numpy())
        else:
          #yPred = nm.model.predict(features['x'], self.batchSize)[:,0]
          yPred = nm.model.predict(features['x'], self.batchSize)
      for t in self.hparam[transform]:
        if t['name'] == 'Sparse':
          yPred = self.PredictTargetX(yPred)
        
      lid.append(yId)
      lpred.append(yPred)
      #batches.append(pa.RecordBatch.from_arrays([pa.array(yId),pa.array(yPred[:,0])], names=names))
      b+=1
      if not b % PROGRESS_INTERVAL:
        print(".", end="")#print(b, yId[0], yPred[:,0][0])#, features['x'][0][0:10])

    #self.prediction = pa.Table.from_arrays([pa.StringArray(lid), pa.array(lpred)], names=names)
    #self.prediction = pa.Table.from_batches(batches)
    print("done")
    self.label = [item.decode("utf-8") for sublist in lid for item in sublist] 
    self.prediction = [item for sublist in lpred for item in sublist] 
    self.dft = pd.DataFrame(self.label, columns = ['id'])
    self.dft['prediction'] = self.prediction    

  def Ensemble(self, ds, model_name, arch, version, predict, predict_transform):
    # Generate predictions for each trial,epoch tuple from best.csv
    #self.LoadBestEpochs(predict)
    self.best = np.ones((41))
    ePred = []
    lid = []
    names = ["id", "prediction"]
    trialModelFile = self.GetTrialModelFile(model_name, version, 1, self.best[0], predict)
    print(trialModelFile)
    # Load the model
    if arch == 'XGB':
      model = XGBRegressor(max_depth=8, learning_rate=7e-3, n_estimators=6000, n_jobs=18, colsample_bytree=0.1)
      model.load_model(trialModelFile)
    elif arch == 'LGB':
      model = lgb.Booster(model_file=trialModelFile)
    else:
      model = tf.keras.models.load_model(trialModelFile)
      if trial == 0:
        model.summary()

    for trial in range(self.best.shape[0]):
      print(trial+1, self.best[trial], end="")
      # Get the model file for this trial and best epoch
       
        
      # Make predictions
      b=0
      lpred = []
      # Inner loop
      # Iterate over each batch in the dataset
      for features in ds:
        # only store the GUIDs once 
        if not trial: 
          yId = self.GetList(features['id'])
          yId = list(yId)
          lid.append(yId)

        if 'XGB' in self.hparam['arch']:
          yPred = model.predict(features['x'].numpy())
        else:
          yPred = model.predict(features['x'], self.batchSize)
        if predict_transform == 'TargetX':
          yhat = self.PredictTargetX(yPred)
          yPred = np.zeros((yPred.shape[0],5))
          yPred[:,0] = yhat[:,0]
        
        lpred.append(yPred)
        b+=1
        if not b % PROGRESS_INTERVAL:
          print(".", end="")
          #print(b, yId[0], yPred[0])#, features['x'][0][0:10])
        
      self.prediction = np.concatenate(lpred)
      ePred.append(self.prediction)
      print(" done")
    
    self.label = [item.decode("utf-8") for sublist in lid for item in sublist] 
    if self.dft is None:
      self.dft = pd.DataFrame(self.label, columns = ['id'])
    if arch == 'TF' or arch == 'AE':
      dim = ePred[0].shape[1]
      if dim > 0:
        for trial in range(len(ePred)):
          for d in range(dim):
            self.dft['trial_%s%d'%(predict, trial*dim+d)] = ePred[trial][:,d]
      else:
        for trial in range(len(ePred)):
          self.dft['trial_%s%d'%(predict, trial)] = ePred[trial]
    else:
      for trial in range(len(ePred)):
        self.dft['trial_%s%d'%(predict, trial)] = ePred[trial]

# %%
AUTOTUNE = tf.data.AUTOTUNE
PROGRESS_INTERVAL = 10


# %%
if IS_JUPYTER:
  sys.argv.append('--epochs')
  sys.argv.append('10')
  sys.argv.append('--target')
  sys.argv.append('0')
  sys.argv.append('--batch_size')
  sys.argv.append('128')
  #sys.argv.append('--steps')
  #sys.argv.append('2940')
  sys.argv.append('--lr')
  sys.argv.append('3e-5')
  sys.argv.append('--decay')
  sys.argv.append('1e-6')
  #sys.argv.append('--epsilon')
  #sys.argv.append('20.0')
  sys.argv.append('--model')
  sys.argv.append('m33')
  sys.argv.append('--arch')
  sys.argv.append('NR')
  sys.argv.append('--monitor')
  sys.argv.append('val_loss')
  sys.argv.append('--version')
  sys.argv.append('8')
  sys.argv.append('--revision')
  sys.argv.append('4')
  sys.argv.append('--trial')
  sys.argv.append('1')
  sys.argv.append('--transform')
  sys.argv.append('NaN,0,0|Slice,0,2304|YX,72,32')
  sys.argv.append('--trainpat')
  sys.argv.append('balance_*.tfr')
  sys.argv.append('--validpat')
  sys.argv.append('balanceval_*.tfr')
  sys.argv.append('--round')
  sys.argv.append('0')
  sys.argv.append('--epoch')
  sys.argv.append('0')
  #sys.argv.append('Slice,0,1536|XY,48,32|Sparse,5')
  #sys.argv.append('NaN,-1,-1|Slice,0,1536|XY,48,32')
  #sys.argv.append('--transform')
  #sys.argv.append('Slice,0,1050')
  #sys.argv.append('--ensemble1')
  #sys.argv.append('diorite,AE,5,breccia,-')
  #sys.argv.append('--ensemble2')
  #sys.argv.append('geode,XGB,2,schist,-')
  #sys.argv.append('--etransform')
  #sys.argv.append('YX,41,5')
  #sys.argv.append('--epsilon')
  #sys.argv.append('0.05')
  #sys.argv.append('--patience')
  #sys.argv.append('5')
  #sys.argv.append('--threshold')
  #sys.argv.append('1e-6')
  sys.argv.append('--train')
  print(sys.argv)


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name")
parser.add_argument("--version", type=int, default=1, help="model version")
parser.add_argument("--round", type=int, default=0, help="model round")
parser.add_argument("--revision", type=int, default=0, help="model revision")
parser.add_argument("--trial", type=int, default=1, help="training trial number")
parser.add_argument("--epoch", type=int, default=0, help="model epoch")
parser.add_argument("--begin", type=int, default=0, help="start with epoch")
parser.add_argument("--patience", type=int, default=0, help="early stopping patience")
parser.add_argument("--threshold", type=float, default=1e-5, help="early stopping threshold")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
parser.add_argument("--epochs", type=int, default=20, help="training epochs")
parser.add_argument("--transform", default='-', help="transform")
parser.add_argument("--arch", default="NR", help="model architecture")
parser.add_argument("--ensemble1", default='-', help="1st ensemble model,arch,version")
parser.add_argument("--etransform", default='-', help="post-ensemble transform")
parser.add_argument("--ensemble2", default='-', help="2nd ensemble model,arch,version")
parser.add_argument("--batch_size", type=int, default=8192, help="batch size")
parser.add_argument("--steps", type=int, default=1000, help="total training steps")
parser.add_argument("--decay_steps", type=int, default=100, help="decay steps")
parser.add_argument("--decay", type=float, default=1e-4, help="decay rate")
parser.add_argument("--target", type=int, default=0, help="target index")
parser.add_argument('--train', action='store_true')
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--loadtrial', action='store_true')
parser.add_argument('--trainera', action='store_true')
parser.add_argument('--datadir', default="./data", help='data directory')
parser.add_argument('--trainpat', default="train*.tfr", help='training file glob pattern')
parser.add_argument('--validpat', default="valid*.tfr", help='validation file glob pattern')
parser.add_argument('--monitor', default="val_loss", help='metric for checkpoint monitor')
parser.add_argument('--loss', default="mse", help='loss function')

if IS_JUPYTER:
  args = parser.parse_args(sys.argv[3:])
else:
  args = parser.parse_args()

print(args)


# %%
napi = NumeraiAPI()

hyperParam = {
  'batch_size': args.batch_size,
}

TARGETIDX = args.target
hyperParam['model'] = args.model
hyperParam['train'] = args.train
hyperParam['round'] = args.round
hyperParam['trial'] = args.trial
hyperParam['revision'] = args.revision
hyperParam['version'] = args.version
hyperParam['epoch'] = args.epoch
hyperParam['epochs'] = args.epochs
hyperParam['begin'] = args.begin
hyperParam['patience'] = args.patience
hyperParam['threshold'] = args.threshold
hyperParam['lr'] = args.lr
hyperParam['decay'] = args.decay
hyperParam['steps'] = args.steps
hyperParam['decay_steps'] = args.decay_steps
hyperParam['epsilon'] = args.epsilon
hyperParam['arch'] = args.arch
hyperParam['loss'] = args.loss
hyperParam['monitor'] = args.monitor
hyperParam['transform'] = []
if not args.transform == '-':
  filters = args.transform.split('|')
  for f in range(len(filters)):
    param = filters[f].split(',')
    if len(param) == 3:
      hyperParam['transform'].append({
        'name': param[0],
        'arg1': param[1],
        'arg2': param[2]
        })
    if len(param) == 4:
      hyperParam['transform'].append({
        'name': param[0],
        'arg1': param[1],
        'arg2': param[2],
        'arg3': param[3]
        })
hyperParam['etransform'] = []
if not args.etransform == '-':
  param = args.etransform.split(',')
  hyperParam['etransform'] = [{
    'name': param[0],
    'arg1': param[1],
    'arg2': param[2]
  }]
hyperParam['ensemble'] = []
if not args.ensemble1 == '-':
  param = args.ensemble1.split(',')
  hyperParam['ensemble'].append( { 'model': param[0], 'arch': param[1], 'version': int(param[2]), 'predict': param[3], 'transform': param[4]} )
if not args.ensemble2 == '-':
  param = args.ensemble2.split(',')
  hyperParam['ensemble'].append( { 'model': param[0], 'arch': param[1], 'version': int(param[2]), 'predict': param[3], 'transform': param[4]} )

print(hyperParam)


# %%
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
  nm = NumeraiModel(args.model, hyperParam)


  # %%
  ds = nm.DataSet(args.datadir, args.trainpat)
  dsv = nm.DataSet(args.datadir, args.validpat)


# %%
with strategy.scope():
  # Train the model
  #if args.trial:
  #  nm.modelTrial = args.trial
  #else:
  #  nm.modelTrial = 1
  nm.modelTrial = 1
  nm.sModelPath = os.path.join(".", nm.GetModelName())
  nm.SetLoss(hyperParam['loss'])
  nm.SetMonitor(hyperParam['monitor'])
  nm.LoadModel()
  nm.SetModelTrial(hyperParam['trial'])
  nm.SetTensorboard(args.tensorboard)
  #nm.modelRevision = napi.GetRound()          
  #autoencoder = nm.model.get_layer('input_output')
  #autoencoder.trainable = False
  #z = nm.model.get_layer('z')
  #z.trainable = False
  nm.model.summary()

# %%
if args.train:
  print("Training")   
  #nm.SetModelTrial(1)
  nm.Train(ds, dsv)
  print("Done")
  path = nm.sModelPath
elif args.trainera:
  for step in range(41):
    trainpat = ''
    validpat = ''
    for era in range(14):
      pat = " train_era%d_*.tfr " % (step*14+era)
      trainpat += pat
      pat = " train_era%d_*.tfr " % ((1+step)*14+era)
      validpat += pat
    trainpat = trainpat[1:]
    validpat = validpat[1:]
    nm.SetModelStep(step+1)
    nm.ResetWeights()
    ds = nm.DataSet(args.datadir, trainpat)
    dsv = nm.DataSet(args.datadir, validpat)
    print("Training step %d" % (step+1))
    
    nm.SetModelTrial(1)
    nm.Train(ds, dsv)

    best = nm.bestEpoch
    print("Removing all but best epoch %d" % (best))
    path = nm.sModelPath
    eraname = '%st%d' %(nm.GetModelFullName(), nm.GetModelTrial())
    cmd = "find %s -depth -type d -name '%s-e*' \! -name '%s-e%d' -exec rm -rf {} \;" % (path, eraname, eraname, best)
    #print(cmd)
    os.system(cmd)
    
  print("Done")

cmd = "ls %s" %(path)
os.system(cmd)

# %%
if False:
  X = ds.take(1)

# %%
if False:
  for ex in X:
    x = ex[0][0]
    y = ex[1][0]
    print(x, y)
    #print(ex)


# %%
if False:
  print(ex[0][0], ex[1][0][0], ex[1][1][0])
  print(ex[0][1], ex[1][0][1], ex[1][1][1])

# %%
#tfr_test_file = "./data/%s_0.tfr"%(hyperParam['model'])
#ensembles=0
#e = hyperParam['ensemble'][0]
#print("Ensemble TFR file %s not detected"%(tfr_test_file))
#print("Building Ensemble Predictions for ", e['model'])
#
#
## %%
#model_name = e['model']
#arch = e['arch']
#version = e['version']
#predict = e['predict']
#predict_transform = e['transform']
#
#
## %%
#nm.best = np.ones((41))
#ePred = []
#lid = []
#names = ["id", "prediction"]
#trialModelFile = self.GetTrialModelFile(model_name, version, 1, self.best[0], predict)
#print(trialModelFile)
#
#
## %%
#model = tf.keras.models.load_model(trialModelFile)
#if trial == 0:
#  model.summary()
#
#trial = 0
#print(trial+1, self.best[trial], end="")
## Get the model file for this trial and best epoch
#             
## Make predictions
#b=0
#lpred = []
## Inner loop
## Iterate over each batch in the dataset
#for features in ds:
#  # only store the GUIDs once 
#  if not trial: 
#    yId = self.GetList(features['id'])
#    yId = list(yId)
#    lid.append(yId)
#  model = nm.Compile(model):
#  nm.Train(model, ds, dsv):
#  model.predict()
#  lpred.append(yPred)
#  b+=1
#  if not b % PROGRESS_INTERVAL:
#    print(".", end="")
#        
#self.prediction = np.concatenate(lpred)
#ePred.append(self.prediction)
#print(" done")
#
## %%
#self.label = [item.decode("utf-8") for sublist in lid for item in sublist] 
#
## %%
#if self.dft is None:
#  self.dft = pd.DataFrame(self.label, columns = ['id'])
#if arch == 'TF' or arch == 'AE':
#  dim = ePred[0].shape[1]
#  if dim > 0:
#    for trial in range(len(ePred)):
#      for d in range(dim):
#        self.dft['trial_%s%d'%(predict, trial*dim+d)] = ePred[trial][:,d]
#  else:
#    for trial in range(len(ePred)):
#      self.dft['trial_%s%d'%(predict, trial)] = ePred[trial]
#  else:
#    for trial in range(len(ePred)):
#      self.dft['trial_%s%d'%(predict, trial)] = ePred[trial]
#
## %%
#tfr_test_file = "./data/%s_0.tfr"%(hyperParam['model'])
#ensembles=0
#for e in hyperParam['ensemble']:
#  if os.path.exists(tfr_test_file):
#    print("Ensemble skipped, TFR file %s detected"%(tfr_test_file))
#  else:
#    print("Building Ensemble Predictions for ", e['model'])
#    nm.Ensemble(ds, e['model'], e['arch'], e['version'], e['predict'], e['transform'])
#  ensembles+=1    
#
## %%
#if ensembles:
#  if not os.path.exists(tfr_test_file):
#    print("Creating TFR files from predictions")
#    trialFeature = [ f for f in nm.dft.columns if f.startswith('trial') ]
#    nm.nFeatures = len(trialFeature)
#    X = nm.dft[trialFeature].to_numpy()
#    Y = nm.label
#    etl = NumeraiETL(".", "./data", hyperParam['model'], shard_size = 50000)
#    etl.Load(X, Y)
#    etl.SaveDataset()
#
#  test_filenames = tf.io.gfile.glob(f"./data/%s_*.tfr"%(hyperParam['model']))
#  TRANSFORM='etransform'
#  ds = nm.GetDataSet(test_filenames, TRANSFORM)
#
## %%
#nm.LoadModel()
#if hyperParam['arch'] == 'TF' or hyperParam['arch'] == 'AE':
#  nm.model.summary()
#
## %%
#nm.Predict(ds, TRANSFORM)
#
## %%
## Save the results for submission to Numerai
#submission_file = nm.SaveSubmissionCSV()
#print(submission_file)
#
#
#
## %%
#napi.Upload(nm.GetModelName(), submission_file)
#
## %%
