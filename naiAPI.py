"""
Example classifier on Numerai data using a lightGBM regression.
To get started, install the required packages: pip install pandas numpy sklearn lightgbm
"""
# !pip install fastparquet numerapi


import csv
import os
from pathlib import Path
from numerapi import NumerAPI

# # Sunshine
# ```
# napi.download_dataset("v4.1/train.parquet", "train.parquet")
# napi.download_dataset("v4.1/validation.parquet", "validation.parquet")
# napi.download_dataset("v4.1/live.parquet", "live.parquet")
# napi.download_dataset("v4.1/live_example_preds.parquet", "live_example_preds.parquet")
# napi.download_dataset("v4.1/validation_example_preds.parquet", "validation_example_preds.parquet")
# napi.download_dataset("v4.1/features.json", "features.json")
# napi.download_dataset("v4.1/meta_model.parquet", "meta_model.parquet")
# ```

# # Rain
# ```
# napi.download_dataset("v4.2/train_int8.parquet", "train_int8.parquet")
# napi.download_dataset("v4.2/validation_int8.parquet", "validation_int8.parquet")
# napi.download_dataset("v4.2/live_int8.parquet", "live_int8.parquet")
# napi.download_dataset("v4.2/live_example_preds.parquet", "live_example_preds.parquet")
# napi.download_dataset("v4.2/validation_example_preds.parquet", "validation_example_preds.parquet")
# napi.download_dataset("v4.2/features.json", "features.json")
# napi.download_dataset("v4.2/meta_model.parquet", "meta_model.parquet")
# ```

# # Atlas
# ```
# version="5.0"
# napi.download_dataset("v%s/features.json" %(version), "features.json")
# napi.download_dataset("v%s/train.parquet" %(version), "train.parquet")
# napi.download_dataset("v%s/validation.parquet" %(version), "validation.parquet")
# napi.download_dataset("v%s/validation_example_preds.parquet" %(version), "validation_example_preds.parquet")
#
# #napi.download_dataset("v%s/live_int8.parquet" %(version), "live_int8.parquet")
# #napi.download_dataset("v%s/live_example_preds.parquet" %(version), "live_example_preds.parquet")
# #napi.download_dataset("v%s/meta_model.parquet" %(version), "meta_model.parquet")
# ```

class NumeraiAPI:
  DR_SUNSHINE = "4.1"
  DR_RAIN = "4.2"
  DR_MIDNIGHT = "4.3"
  DR_ATLAS = "5.0"
  LIVEDATA = "live.parquet"
  TRAINDATA = "train.parquet"
  VALIDDATA = "validation.parquet"
  METAMODEL = "meta.parquet"
  FEATURES = "features.json"

  def __init__(self):
    self.publicID = os.getenv("NAI_IDENTITY")
    self.secretKey = os.getenv("NAI_SECRET")
    self.submission = {}
    # Default to RAIN values
    self.version = self.DR_ATLAS
    self.archive="parquet"
    self.type=""
    self.trainPrefix="train"
    self.validPrefix="validation"
    self.livePrefix="live"
    self.metaPrefix="meta_model"
    self.featuresPrefix="features"
    self.trainSuffix=self.type
    self.validSuffix=self.type
    self.liveSuffix=self.type
    self.metaSuffix=""
    self.featuresSuffix="json"
    self.Session()

  def Session(self):
    self.napi = NumerAPI(secret_key=self.secretKey, public_id = self.publicID, verbosity="info")
    self.GetModels()
    self.GetRound()

  def SetKey(self, identity, secret):
    self.publicID = identity
    self.secretKey = secret
    
  def SetVersion(self, version=None):
    if version == self.DR_SUNSHINE:
      self.type=""
    elif version == self.DR_RAIN:
      self.type="_int8"
    elif version == self.DR_MIDNIGHT:
      self.type="_int8"
    elif version == self.DR_ATLAS:
      self.type=""
    else:
      return
    self.trainSuffix=self.type
    self.validSuffix=self.type
    self.liveSuffix=self.type
    self.version = version

  def SetTrainPrefix(self, prefix):
    self.trainPrefix = prefix

  def SetValidPrefix(self, prefix):
    self.validPrefix = prefix

  def SetLivePrefix(self, prefix):
    self.livePrefix = prefix

  def SetTrainSuffix(self, suffix):
    self.trainSuffix = suffix

  def SetValidSuffix(self, suffix):
    self.validSuffix = suffix

  def SetLiveSuffix(self, suffix):
    self.liveSuffix = suffix

  def TrainDownloadFile(self):
    fname = "v%s/%s%s.parquet" % (self.version, self.trainPrefix, self.trainSuffix)
    return fname
    
  def ValidDownloadFile(self):
    fname = "v%s/%s%s.parquet" % (self.version, self.validPrefix, self.validSuffix)
    return fname
    
  def LiveDownloadFile(self):
    fname = "v%s/%s%s.parquet" % (self.version, self.livePrefix, self.liveSuffix)
    return fname
    
  def MetaDownloadFile(self):
    fname = "v%s/%s.parquet" % (self.version, self.metaPrefix)
    return fname
   
  def FeaturesDownloadFile(self):
    fname = "v%s/%s.%s" % (self.version, self.featuresPrefix, self.featuresSuffix)
    return fname
   
  def DownloadFeatures(self, verbose=False):
    if verbose:
      print('downloading features: %s as %s' % (self.FeaturesDownloadFile(), self.FEATURES))
    self.napi.download_dataset(self.FeaturesDownloadFile(), self.FEATURES)
    if verbose:
      print("download complete")

  def DownloadTrain(self, verbose=False):
    if verbose:
      print('downloading training data: %s as %s' % (self.TrainDownloadFile(), self.TRAINDATA))
    self.napi.download_dataset(self.TrainDownloadFile(), self.TRAINDATA)
    if verbose:
      print("download complete")

  def DownloadValid(self, verbose=False):
    if verbose:
      print('downloading validation data: %s as %s' % (self.ValidDownloadFile(), self.VALIDDATA))
    self.napi.download_dataset(self.ValidDownloadFile(), self.VALIDDATA)
    if verbose:
      print("download complete")

  def DownloadLive(self, verbose=False):
    if verbose:
      print('downloading live data: %s as %s' % (self.LiveDownloadFile(), self.LIVEDATA))
    self.napi.download_dataset(self.LiveDownloadFile(), self.LIVEDATA)
    if verbose:
      print("download complete")

  def DownloadMetaModel(self, verbose=False):
    if verbose:
      print('downloading meta model: %s as %s' % (self.MetaDownloadFile(), self.METAMODEL))
    self.napi.download_dataset(self.MetaDownloadFile(), self.METAMODEL)
    if verbose:
      print("download complete")

  def ListData(self):
    dataset = self.napi.list_datasets(self.GetRound())
    for d in dataset:
      print(d)

  def DownloadAll(self):
    self.DownloadTrain()
    self.DownloadValid()
    self.DownloadLive()
    self.DownloadMetaModel()

  def GetDatasets(self):
    self.datasets = self.napi.list_datasets()
    return self.datasets

  def GetRound(self):
    self.thisRound = self.napi.get_current_round()
    return self.thisRound

  def GetModels(self):
    self.model = self.napi.get_models()

  def Upload(self, modelName, filePath):
    modelID = self.model[modelName] 
    self.submission[modelName] = self.napi.upload_predictions(file_path=filePath, model_id=modelID)
    self.napi.submission_status(model_id=modelID)
    return self.submission[modelName]


