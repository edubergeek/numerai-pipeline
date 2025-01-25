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
from tensorflow.keras import backend as K
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# %%
from numerapi import NumerAPI
from naiAPI import NumeraiAPI
from naiETL import _floatvector_feature, _float_feature, _int64_feature, _bytes_feature, _dtype_feature, TrainingSet, ETL, NumeraiETL

# %%
TRANSFORM = 'transform'
def decode_tfr(record_bytes):
    schema =  {
      "era": tf.io.FixedLenFeature([], dtype=tf.int64),
      "id": tf.io.FixedLenFeature([], dtype=tf.string),
      "x":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
      "y":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
    }    
    example = tf.io.parse_single_example(record_bytes, schema)
    return example

def reshape_C6(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'C6':
        # clone the last 50 values 4 times 50 50 50 50
        temp1 = tf.slice(features['x'], [0], [1000])
        temp2 = tf.slice(features['x'], [1000], [50])
        temp3 = tf.repeat(temp2, 4)
        temp4 = tf.concat([temp1, temp3], 0)
        features['x'] = temp4
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features['x'] = tf.reshape(features['x'], [6, yDim, xDim])
        features['x'] = tf.transpose(features['x'], perm=[1, 2, 0])        
    return features

def reshape_ZeroPad(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'ZP':
        # parse a set of feature fragments into an MxN matrix of features
        # arg1 stores M:N
        # arg2 stores the fragment offsets separated by colon
        # 0:35:325:460:511:836:976:1121:1216:1506:1796:2086:2376
        # zero padding dimension N
        arg1 = t['arg1']
        arg2 = t['arg2']
        parts = arg1.split(':')
        yDim = int(parts[0])
        xDim = int(parts[1])
        parts = arg2.split(':')
        for p in range(len(parts)-1):
          start = int(parts[p])
          end = int(parts[p+1])
          xlen = end - start
          x = tf.slice(features['x'], [start], [xlen])
          zeropad = tf.constant(0.0, dtype=tf.float32)
          padlen = xDim - xlen
          #print("pad length",p,"=",padlen)  
          pad = tf.repeat(zeropad, padlen)
          if p == 0:
            X = tf.concat([x, pad], 0)
          else:
            X = tf.concat([X, x, pad], 0)
        features['x'] = tf.reshape(X, [yDim, xDim])
        #features = tf.transpose(features, perm=[1, 2, 0])
    return features

def reshape_YXZ(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'YXZ':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        zDim = int(t['arg3'])        
        features['x'] = tf.reshape(features['x'], [yDim, xDim, zDim])
    return features

def reshape_XY(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'XY':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features['x'] = tf.reshape(features['x'], [xDim, yDim])        
        features['x'] = tf.transpose(features['x'])
    return features

def reshape_YX(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'YX':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features['x'] = tf.reshape(features['x'], [yDim, xDim])
    return features

def reshape_Slice(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Slice':
        fromDim = int(t['arg1'])
        toDim = int(t['arg2'])
        features['x'] = tf.slice(features['x'], [fromDim], [toDim])
    return features

def reshape_Even(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Even':
        fromDim = int(t['arg1'])
        toDim = int(t['arg2'])
        features['x'] = tf.gather(features['x'], tf.constant(np.arange(fromDim, toDim*2, 2)))
        features['x'] = tf.slice(features['x'], [fromDim], [toDim])
    return features

def replace_NaN(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'NaN':
        featureVal = float(t['arg1'])
        mask = tf.math.is_nan(features['x'])
        maskVal = tf.cast(tf.ones_like(mask), tf.float32) * tf.constant(featureVal, dtype=tf.float32)
        features['x'] = tf.where(mask, maskVal, features['x'])
    return features


def reshape_Pad(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Pad':
        fromPos = int(t['arg1'])
        toPos = int(t['arg2'])
        featureVal = float(t['arg3'])
        # initialize an array of "featureVal" values in the correct shape
        #val = np.ones((toPos-fromPos)) * featureVal
        #pad = tf.constant(val, dtype=tf.float32)
        pad = tf.slice(features['x'], [0], [toPos-fromPos])
        features['x'] = tf.concat([features['x'], pad], 0)
        
    return features

# %%
def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# %%
class NumeraiModel():
  def __init__(self, modelname, modelpath, hyperParam):
    #self.sModelPath = "../round/%d" %(hyperParam["round"])
    self.sModelPath = "../%s/%s" %(modelpath, modelname)
    self.sModelName = modelname
    self.modelVersion = hyperParam["version"]
    self.modelRevision = hyperParam["revision"]
    self.modelTrial = hyperParam["trial"]
    self.modelEpoch = hyperParam["epoch"]
    self.sModelSuffix = ""    
    self.batchSize = hyperParam["batch_size"]
    self.targetX = [0.1, 0.48, 0.0, 0.0, 0.0]
    self.dft = None

    self.hparam = hyperParam

  def SetModelRevision(self, revision):
    self.modelRevision = revision

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
    return "%sv%dr%d" %(self.sModelName, self.modelVersion, self.modelRevision)

  def GetModelFullName(self):
    return "%sv%dr%d" %(self.sModelName, self.modelVersion, self.modelRevision)

  def GetModelName(self):
    return self.sModelName

  def GetList(self, f):
    return f.numpy().tolist()

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
        if t['name'] == 'ZP':
          dataset = dataset.map(reshape_ZeroPad, num_parallel_calls=at)
        if t['name'] == 'XY':
          dataset = dataset.map(reshape_XY, num_parallel_calls=at)
        if t['name'] == 'YX':
          dataset = dataset.map(reshape_YX, num_parallel_calls=at)
        if t['name'] == 'YXZ':
          dataset = dataset.map(reshape_YXZ, num_parallel_calls=at)
        if t['name'] == 'Even':
          dataset = dataset.map(reshape_Even, num_parallel_calls=at)
        if t['name'] == 'Slice':
          dataset = dataset.map(reshape_Slice, num_parallel_calls=at)
        if t['name'] == 'NaN':
          dataset = dataset.map(replace_NaN, num_parallel_calls=at)
        if t['name'] == 'Pad':
          dataset = dataset.map(reshape_Pad, num_parallel_calls=at)
    
    dataset = dataset.batch(self.batchSize).prefetch(at).repeat(count=1)

    return dataset
       
  def SaveSubmissionCSV(self):
    # Save predictions as a CSV and upload to https://numer.ai
    submissionCSV = self.GetModelFullName() + ".csv"
    print(self.dft['prediction'].shape)
    if self.hparam['arch'] == 'NC':
      print(self.dft['prediction'][0:5])
    #if (self.hparam['arch'] == 'AE' or self.hparam['arch'] == 'NR' or self.hparam['arch'] == 'NC') and self.dft['prediction'].ndim == 1:
    if (self.hparam['arch'] == 'NC') and self.dft['prediction'].ndim == 1:
      pred = [x for row in self.dft['prediction'] for x in row]
      #pred = self.dft['prediction'].to_numpy()
      self.dft['prediction'] = pred
    if self.hparam['arch'] == 'NC':
      print(self.dft['prediction'][0:5])
    self.dft.to_csv(submissionCSV, header=True, index=False)
    return submissionCSV

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
        _, yPred = self.model.predict(features['x'], self.batchSize)
      elif self.hparam['arch'] == 'XGB' or self.hparam['arch'] == 'LGB':
        yPred = self.model.predict(features['x'].numpy())
      elif self.hparam['arch'] == 'NC':
        yPred = self.model.predict(features['x'], self.batchSize)
      else:
        yPred = self.model.predict(features['x'], self.batchSize)[:,0]
          #yPred = nm.model.predict(features['x'], self.batchSize)
      for t in self.hparam[transform]:
        if t['name'] == 'Sparse':
          print(yPred.shape)
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
    # clamp values to range 0 to 1.0
    self.prediction = np.asarray(self.prediction)
    self.prediction[self.prediction < 0.0] = 0.0
    self.prediction[self.prediction > 1.0] = 1.0
    self.dft = pd.DataFrame(self.label, columns = ['id'])
    self.dft['prediction'] = self.prediction    

  def Ensemble(self, ds, model_name, arch, version, predict, predict_transform):
    # Generate predictions for each trial,epoch tuple from best.csv
    self.LoadBestEpochs(predict)
    ePred = []
    lid = []
    names = ["id", "prediction"]
    for trial in range(self.best.shape[0]):
      print(trial+1, self.best[trial], end="")
      # Get the model file for this trial and best epoch
      trialModelFile = self.GetTrialModelFile(model_name, version, trial+1, self.best[trial], predict)
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

        if arch == 'XGB' or arch == 'LGB':
          yPred = model.predict(features['x'].numpy())
        elif arch == 'AE':
          _, yPred = model.predict(features['x'], self.batchSize)
        else:
          yPred = model.predict(features['x'], self.batchSize)
        if predict_transform == 'Sparse':
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
    if arch == 'NC' or arch == 'NR' or arch == 'AE':
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
IS_JUPYTER = False
if IS_JUPYTER:
  sys.argv.append('--model')
  sys.argv.append('pumice')
  sys.argv.append('--arch')
  sys.argv.append('LGB')
  sys.argv.append('--version')
  sys.argv.append('2')
  sys.argv.append('--round')
  sys.argv.append('0')
  sys.argv.append('--revision')
  sys.argv.append('434')
  sys.argv.append('--trial')
  sys.argv.append('1')
  sys.argv.append('--epoch')
  sys.argv.append('0')
  #sys.argv.append('--batch_size')
  #sys.argv.append('1024')
  #sys.argv.append('--filepat')
  #sys.argv.append('./data/test_*.tfr')   
  sys.argv.append('--transform')
  sys.argv.append('NaN,-1,-1')
  #sys.argv.append('--transform')
  #sys.argv.append('Even,0,525|XY,25,21')
  #sys.argv.append('--ensemble1')
  #sys.argv.append('granite,TF,7,gneiss')
  #sys.argv.append('--ensemble2')
  #sys.argv.append('geode,XGB,2,schist')
  #sys.argv.append('--etransform')
  #sys.argv.append('YX,41,5')


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name")
parser.add_argument("--path", default="xyzzy", help="model path")
parser.add_argument("--version", type=int, help="model version")
parser.add_argument("--round", type=int, help="model round")
parser.add_argument("--revision", type=int, help="model revision")
parser.add_argument("--trial", type=int, help="model trial")
parser.add_argument("--epoch", type=int, help="model epoch")
parser.add_argument("--transform", default='-', help="transform")
parser.add_argument("--arch", default="TF", help="model architecture")
parser.add_argument("--ensemble1", default='-', help="1st ensemble model,arch,version")
parser.add_argument("--etransform", default='-', help="post-ensemble transform")
parser.add_argument("--ensemble2", default='-', help="2nd ensemble model,arch,version")
parser.add_argument("--filepat", default="./data/live_*.tfr", help="prediction files")
parser.add_argument("--batch_size", type=int, default=8192, help="batch size")

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

hyperParam['model'] = args.model
hyperParam['round'] = args.round
hyperParam['revision'] = args.revision
hyperParam['trial'] = args.trial
hyperParam['version'] = args.version
hyperParam['epoch'] = args.epoch
hyperParam['arch'] = args.arch
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
  hyperParam['ensemble'].append( { 'model': param[0], 'arch': param[1], 'version': param[2], 'predict': param[3], 'transform': param[4]} )
if not args.ensemble2 == '-':
  param = args.ensemble2.split(',')
  hyperParam['ensemble'].append( { 'model': param[0], 'arch': param[1], 'version': param[2], 'predict': param[3], 'transform': param[4]} )

print(hyperParam)


# %%
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
  nm = NumeraiModel(args.model, args.path, hyperParam)


# %%
test_filenames = tf.io.gfile.glob(args.filepat)
TRANSFORM='transform'
ds = nm.GetDataSet(test_filenames, TRANSFORM)


# %%
modelFile = nm.GetModelFile()
print("Loading model %s ..."%(modelFile))
if hyperParam['arch'] == 'XGB':
  nm.model = XGBRegressor(max_depth=8, learning_rate=7e-3, n_estimators=6000, n_jobs=18, colsample_bytree=0.1)
  nm.model.load_model(modelFile)
elif hyperParam['arch'] == 'LGB':
  nm.model = lgb.Booster(model_file=modelFile)
else:
  nm.model = tf.keras.models.load_model(modelFile, custom_objects={'z': sample})

# %%
napi = NumeraiAPI()
nm.SetModelRevision(napi.GetRound())

# %%
#nm.LoadModel()
if hyperParam['arch'] == 'NR' or hyperParam['arch'] == 'NC' or hyperParam['arch'] == 'AE':
  nm.model.summary()

# %%
nm.Predict(ds, TRANSFORM)

# %%
# Save the results for submission to Numerai
submission_file = nm.SaveSubmissionCSV()
print(submission_file)



# %%
# !wc -l pumicev2r444.csv

# %%
napi.Upload(nm.GetModelName(), submission_file)

# %%
# !ls -lt andesite*|head

# %%
nm.dft.describe()

# %%
nm.dft['prediction'].shape

# %%
pred = nm.dft['prediction'].to_numpy()

# %%
pred.shape

# %%
pred

# %%
