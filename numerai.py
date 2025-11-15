#!/usr/bin/env python
"""
Example classifier on Numerai data using a xgboost regression.
To get started, install the required packages: pip install pandas numpy sklearn xgboost
"""

import csv
import os
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime

""" 
Numerai base class
Read training and tournament data in CSV format
Write submission data in CSV format
Scoring and evaluation methods
Training and validation methods
"""
class Numerai:
  def __init__(self):
    self.keyTarget = f"target"
    self.keyPrediction = f"prediction"
    self.keyValidation = f"validation"
    self.keyExamplePreds = f"example_preds"
    self.keyEra = f"era"
    self.keyFeature = f"feature"
    self.keyTarget = f"target"
    self.sModelPath = f"./"
    self.sModelName = f"numerai"
    self.modelVersion = 1
    self.modelRevision = 0
    self.modelTrial = 1
    self.modelEpoch = 0
    self.sModelSuffix = f""
    self.isTraining = False
    dt = datetime.now()
    self.isLive = dt.weekday() < 5

  def SetModelPath(self, path):
    self.sModelPath = path
    if not os.path.exists(path):
      os.makedirs(path)

  def GetModelPath(self):
    return self.sModelPath

  def SetModelName(self, name):
    self.sModelName = name

  def SetModelVersion(self, version):
    self.modelVersion = version

  def GetModelVersion(self):
    return self.modelVersion

  def SetModelRevision(self, revision):
    self.modelRevision = revision

  def GetModelRevision(self):
    return self.modelRevision

  def SetModelTrial(self, trial):
    self.modelTrial = trial

  def GetModelTrial(self):
    return self.modelTrial

  def SetModelEpoch(self, epoch):
    self.modelEpoch = epoch

  def SetModelSuffix(self, suffix):
    self.sModelSuffix = suffix

  def GetModelName(self):
    return self.sModelName

  def GetModelFullName(self):
    return "%sv%dr%d" %(self.sModelName, self.modelVersion, self.modelRevision)

  def GetModelFile(self):
    return "%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, self.sModelName, self.modelVersion, self.modelRevision, self.modelTrial, self.modelEpoch, self.sModelSuffix)

  def GetTrain(self):
    return self.dfTrain

  def SetTrain(self, df):
    self.dfTrain = df
    self.vkeyFeature = [
        f for f in df.columns if f.startswith(self.keyFeature)
    ]
    self.nFeatures = len(self.vkeyFeature)
    self.vkeyTarget = [
        t for t in df.columns if t.startswith(self.keyTarget)
    ]
    self.nTargets = len(self.vkeyTarget)
    self.dfTrain["erano"] = self.dfTrain.era
    self.eras = self.dfTrain.erano.astype(int) 
    self.nTrain = len(self.dfTrain)
    self.nEra = len(self.eras)

  def GetNTrain(self):
    return self.nTrain

  def GetNEra(self):
    return self.nEra

  def GetTrainX(self):
    x=self.dfTrain[self.GetFeatureKeys()].to_numpy()
    return x

  def GetTrainY(self):
    y = self.dfTrain[self.keyTarget].to_numpy()
    return y

  def GetValidX(self):
    x=self.dfValid[self.GetFeatureKeys()].to_numpy()
    return x

  def GetValidY(self):
    y = self.dfValid[self.keyTarget].to_numpy()
    return y

  def SetValid(self, df):
    self.dfValid = df
    self.nValid = len(self.dfValid)

  def GetNValid(self):
    return self.nValid

  def GetTournament(self):
    return self.dfTournament

  def GetValid(self):
    return self.dfValid

  def SetTournament(self, df):
    self.dfTournament = df
    self.nTournament = len(df)

    # get validation subset of tournament data
    #self.dfValid = df[df.data_type == self.keyValidation]
    #self.nValid = len(self.dfValid)

  def GetNTournament(self):
    return self.nTournament

  def GetNFeatures(self):
    return self.nFeatures

  def GetFeatureKeys(self):
    return self.vkeyFeature

  def GetValidFeatureExposures(self):
    return self.dfValid[self.vkeyFeature].apply(lambda d: self.Correlation(self.dfValid[self.keyPrediction], d), axis=0)

  def ValidFeatureExposure(self):
    fe = self.GetValidFeatureExposures()
    fe_era_max = self.dfValid.groupby(self.keyEra).apply(lambda d: d[self.vkeyFeature].corrwith(d[self.keyPrediction]).abs().max())
    fe_max = fe_era_max.mean()
    return fe, fe_max

  def Correlation(self, predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    # Submissions are scored by spearman correlation
    self.corrcoef= np.corrcoef(ranked_preds, targets)[0, 1]
    return self.corrcoef

  def GetTrainCorrelation(self):
    return self.Correlation(self.dfTrain[self.keyPrediction], self.dfTrain[self.keyTarget])

  def GetValidCorrelation(self):
    return self.Correlation(self.dfValid[self.keyPrediction], self.dfValid[self.keyTarget])

  # convenience method for scoring
  def Score(self, df):
    return self.Correlation(df[self.keyPrediction], df[self.keyTarget])

  # Payout is just the score clipped at +/-25%
  def Payout(self, scores):
    return scores.clip(lower=-0.25, upper=0.25)

  def TrainPayout(self):
    return self.Payout(self.CorrelationTrain()).mean()

  def ValidPayout(self):
    return self.Payout(self.corrValid).mean()

  # Check the per-era correlations on the training set (in sample)
  def CorrelationTrain(self):
    return self.dfTrain.groupby(self.keyEra).apply(self.Score)

  # Check the per-era correlations on the validation set (out of sample)
  def CorrelationValid(self):
    self.corrValid = self.dfValid.groupby(self.keyEra).apply(self.Score)
    self.corrmean = self.corrValid.mean()
    self.corrstd = self.corrValid.std(ddof=0)
    # Check the "sharpe" ratio on the validation set
    self.sharpeValid = self.corrmean / self.corrstd
    return self.corrValid

  def Validate(self):
    self.CorrelationValid()
    rolling_max = (self.corrValid + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (self.corrValid + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    _, fe_max = self.ValidFeatureExposure()
    fnm = self.ValidFeatureNeutralMean()
    return self.corrValid, self.corrmean, self.corrstd, self.sharpeValid, max_drawdown, fe_max, fnm
    
  def GetValidCorrMean(self):
    return self.corrmean

  def GetValidCorrStd(self):
    return self.corrstd

  def GetValidSharpe(self):
    return self.sharpeValid

  def Evaluate(self):
    # Check the per-era correlations on the training set (in sample)
    corrcoef = self.CorrelationTrain()
    print(f"On training the correlation has mean {corrcoef.mean()} and std {corrcoef.std(ddof=0)}")
    print(f"On training the average per-era payout is {self.TrainPayout()}")

    """Validation Metrics"""
    # Check the per-era correlations on the validation set (out of sample)
    corrcoef, corrmean, corrstd, sharpe, max_drawdown, fe_max, fnm = self.Validate()
    print(f"On validation the correlation has mean {corrmean} and std {corrstd}")
    print(f"Validation Sharpe: {sharpe}")
    print(f"On validation the average per-era payout is {self.ValidPayout()}")
    print(f"Downside risk:")
    print(f"max drawdown: {max_drawdown}")
    print(f"Max Feature Exposure: {fe_max}")
    print(f"Feature Neutral Mean is {fnm}")

    print("calculating MMC stats...")
    # MMC over validation depends on LoadPredictions
    mmc_mean, mmc_sharpe, mmc_diff = self.ValidMMC()
    print(
        f"MMC Mean: {mmc_mean}\n"
        f"Corr Plus MMC Sharpe:{mmc_sharpe}\n"
        f"Corr Plus MMC Diff:{mmc_diff}"
    )

    # compute correlation of validation predictions and example predictions
    # depends on LoadPredictions
    corr_with_example_preds = self.FullCorrelation()
    print(f"Corr with example preds: {corr_with_example_preds}")

  def TrainMetrics(self):
    tcorr = self.CorrelationTrain()
    print(tcorr.mean())
    print(tcorr.std(ddof=0))
    print(self.TrainPayout())

  def ValidMetrics(self):
    vcorr, vcorr_mean, vcorr_std, vsharpe, drawdown, fe, fnm = self.Validate()
    print(vcorr_mean)
    print(vcorr_std)
    print(self.ValidPayout())
    print(vsharpe)
    print(drawdown)
    print(fe)
    print(fnm)
    mmc_mean, mmc_sharpe, mmc_diff = self.ValidMMC()
    print(mmc_mean)
    print(mmc_diff)
    print(mmc_sharpe)
    corr_with_example_preds = self.FullCorrelation()
    print(corr_with_example_preds)

  def Metrics(self):
    self.TrainMetrics()
    self.ValidMetrics()

  # Read the csv file into a pandas Dataframe as float16 to save space
  def ReadParquet(self, file_path):
    #with open(file_path, 'r') as f:
    #  column_names = next(csv.reader(f))

    #dtypes = {x: np.float32 for x in column_names if x.startswith((self.keyFeature, self.keyTarget))}
    df = pd.read_parquet(file_path)

    return df

  # Read the csv file into a pandas Dataframe as float16 to save space
  def ReadCSV(self, file_path):
    with open(file_path, 'r') as f:
      column_names = next(csv.reader(f))

    dtypes = {x: np.float32 for x in column_names if x.startswith((self.keyFeature, self.keyTarget))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)

    return df

  def ReadTrain(self):
    return self.ReadParquet('train.parquet')

  def ReadValid(self):
    return self.ReadParquet('validation.parquet')

  def ReadTournament(self):
    return self.ReadParquet('tournament.parquet')

  def ReadLive(self):
    return self.ReadParquet('live.parquet')

  def SaveTestCSV(self):
    # Save predictions as a CSV and upload to https://numer.ai
    testCSV = self.GetModelFullName() + "_test.csv"
    self.dfValid[self.keyPrediction].to_csv(testCSV, header=True)
    return testCSV

  def SaveSubmissionCSV(self):
    # Save predictions as a CSV and upload to https://numer.ai
    submissionCSV = self.GetModelFullName() + ".csv"
    self.dfTournament[self.keyPrediction].to_csv(submissionCSV, header=True)
    return submissionCSV

  def LoadData(self):
    dtr = self.ReadTrain()
    self.SetTrain(dtr)
    #dtr[self.vkeyFeature] = dtr[self.vkeyFeature].fillna(dtr[self.vkeyFeature].median(skipna=True))
    dtr[self.vkeyFeature] = dtr[self.vkeyFeature].fillna(0.5)
    dva = self.ReadValid()
    self.SetValid(dva)
    dva[self.vkeyFeature] = dva[self.vkeyFeature].fillna(0.5)
    dva[self.keyTarget] = dva[self.keyTarget].fillna(0.5)
    # after replacing NaN features drop any target NaN validation rows
    #dva = dva.dropna()
    dto = self.ReadLive()
    self.SetTournament(dto)
    dto[self.vkeyFeature] = dto[self.vkeyFeature].fillna(0.5)
    return (dtr, dva, dto)

  def LoadLive(self):
    dtr = self.ReadTrain()
    self.SetTrain(dtr)
    dto = self.ReadLive()
    self.SetTournament(dto)
    dto[self.vkeyFeature] = dto[features].fillna(0.5)
    return (dto)

  def MergeData(self, split = 0.5):
    dtr = self.GetTrain()
    ntr = self.GetNTrain()
    dva = self.GetValid()
    nva = self.GetNValid()
    nvat = int(nva * split)
    nvav = nva - nvat
    dvat = dva.head(nvat)
    dvav = dva.tail(nvav)
    dva = []

    dtr = pd.concat([dtr, dvat], axis=0)
    self.SetTrain(dtr)
    self.SetValid(dvav)

  def FilterTrain(self, era=4):
    df = self.GetTrain()
    dfFilt = df[(self.eras % era == 0)]
    self.SetTrain(dfFilt)

  def LoadPredictions(self):
    example_preds = pd.read_csv("example_predictions.csv").set_index("id")[self.keyPrediction]
    validation_example_preds = example_preds.loc[self.dfValid.index]
    self.dfFull = pd.concat([validation_example_preds, self.dfValid[self.keyPrediction], self.dfValid[self.keyEra]], axis=1)
    self.dfFull.columns = [self.keyExamplePreds, self.keyPrediction, self.keyEra]
    self.dfValid[self.keyExamplePreds] = validation_example_preds

  def ValidMMC(self):
    # Load example preds to get MMC metrics
    self.LoadPredictions()

    mmc_scores = []
    corr_scores = []
    for _, x in self.dfValid.groupby(self.keyEra):
      series = self.NeutralizeSeries(pd.Series(self.Unif(x[self.keyPrediction])), pd.Series(self.Unif(x[self.keyExamplePreds])))
      mmc_scores.append(np.cov(series, x[self.keyTarget])[0, 1] / (0.29 ** 2))
      corr_scores.append(self.Correlation(self.Unif(x[self.keyPrediction]), x[self.keyTarget]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
    corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - self.sharpeValid
    return val_mmc_mean, corr_plus_mmc_sharpe, corr_plus_mmc_sharpe_diff

  def FullCorrelation(self):
    # Check correlation with example predictions
    per_era_corrs = self.dfFull.groupby(self.keyEra).apply(lambda d: self.Correlation(self.Unif(d[self.keyPrediction]), self.Unif(d[self.keyExamplePreds])))
    return per_era_corrs.mean()

  # to neutralize a column in a df by many other columns on a per-era basis
  def Neutralize(self, df, columns, extra_neutralizers=None, proportion=1.0, normalize=True, era_col='era'):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
      extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
      print(u, end="\r")
      df_era = df[df[era_col] == u]
      scores = df_era[columns].values
      if normalize:
        scores2 = []
        for x in scores.T:
          x = (pd.Series(x).rank(method="first").values - .5) / len(x)
          scores2.append(x)
          scores = np.array(scores2).T
          extra = df_era[extra_neutralizers].values
          exposures = np.concatenate([extra], axis=1)
      else:
        exposures = df_era[extra_neutralizers].values

      scores -= proportion * exposures.dot(
        np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))
      scores /= scores.std(ddof=0)
      computed.append(scores)

    return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)

  # to neutralize any series by any other series
  def NeutralizeSeries(self, series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack( (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot( np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized

  def Unif(self, df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


  def GetFeatureNeutralMean(self, df):
    feature_cols = [c for c in df.columns if c.startswith(self.keyFeature)]
    df.loc[:, "neutral_sub"] = self.Neutralize(df, [self.keyPrediction], feature_cols)[self.keyPrediction]
    scores = df.groupby(self.keyEra).apply( lambda x: self.Correlation(x["neutral_sub"], x[self.keyTarget])).mean()
    return np.mean(scores)

  def ValidFeatureNeutralMean(self):
    return self.GetFeatureNeutralMean(self.dfValid)


def main():
    nai = Numerai()

    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data = nai.ReadCSV("numerai_training_data.csv")
    training_data.head()
    nai.SetTrain(training_data)
    print(f"Loaded {nai.GetNTrain()} Training rows")

    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = nai.ReadCSV("numerai_tournament_data.csv")
    nai.SetTournament(tournament_data)
    print(f"Loaded {nai.GetNTournament()} Tournament rows")
    print(f"Loaded {nai.GetNValid()} Validation rows")

    print(f"Loaded {nai.GetNFeatures()} features")

if __name__ == '__main__':
    main()


def bumbai():
    print("Generating predictions...")
    training_data[PREDICTION_NAME] = model.predict(training_data[feature_names])
    tournament_data[PREDICTION_NAME] = model.predict(tournament_data[feature_names])

    # Check the per-era correlations on the training set (in sample)
    train_correlations = training_data.groupby("era").apply(score)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")

    """Validation Metrics"""
    # Check the per-era correlations on the validation set (out of sample)
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and "
          f"std {validation_correlations.std(ddof=0)}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

    # Check the "sharpe" ratio on the validation set
    validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
    print(f"Validation Sharpe: {validation_sharpe}")

    print("checking max drawdown...")
    rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                                  min_periods=1).max()
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    print(f"max drawdown: {max_drawdown}")

    # Check the feature exposure of your validation predictions
    feature_exposures = validation_data[feature_names].apply(lambda d: correlation(validation_data[PREDICTION_NAME], d),
                                                             axis=0)
    max_per_era = validation_data.groupby("era").apply(
        lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
    max_feature_exposure = max_per_era.mean()
    print(f"Max Feature Exposure: {max_feature_exposure}")

    # Check feature neutral mean
    print("Calculating feature neutral mean...")
    feature_neutral_mean = get_feature_neutral_mean(validation_data)
    print(f"Feature Neutral Mean is {feature_neutral_mean}")

    # Load example preds to get MMC metrics
    example_preds = pd.read_csv("example_predictions.csv").set_index("id")["prediction"]
    validation_example_preds = example_preds.loc[validation_data.index]
    validation_data[self.keyExamplePreds] = validation_example_preds

    print("calculating MMC stats...")
    # MMC over validation
    mmc_scores = []
    corr_scores = []
    for _, x in validation_data.groupby("era"):
        series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),
                                   pd.Series(unif(x["ExamplePreds"])))
        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))
        corr_scores.append(correlation(unif(x[PREDICTION_NAME]), x[TARGET_NAME]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
    corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

    print(
        f"MMC Mean: {val_mmc_mean}\n"
        f"Corr Plus MMC Sharpe:{corr_plus_mmc_sharpe}\n"
        f"Corr Plus MMC Diff:{corr_plus_mmc_sharpe_diff}"
    )

    # Check correlation with example predictions
    full_df = pd.concat([validation_example_preds, validation_data[PREDICTION_NAME], validation_data["era"]], axis=1)
    full_df.columns = ["example_preds", "prediction", "era"]
    per_era_corrs = full_df.groupby('era').apply(lambda d: correlation(unif(d["prediction"]), unif(d["example_preds"])))
    corr_with_example_preds = per_era_corrs.mean()
    print(f"Corr with example preds: {corr_with_example_preds}")

    # Save predictions as a CSV and upload to https://numer.ai
    tournament_data[PREDICTION_NAME].to_csv("submission.csv", header=True)


