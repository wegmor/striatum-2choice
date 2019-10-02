import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import h5py
import pathlib

from utils import readSessions
#import analysisTunings, analysisDecoding

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
cacheFolder =  pathlib.Path("cache")

cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisTunings.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)

tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005

#%% Fraction of tuned neurons
rewardTuningData = tuningData[tuningData.action.isin(["pL2Cr", "pR2Cr"])]
rewardTuned = rewardTuningData.set_index(["animal", "date", "neuron", "action"]).signp.unstack()
fracReward = rewardTuned.any(axis=1).mean(axis=0)
print("Fraction of neurons positively tuned to rewards: {:.3f}%".format(fracReward*100))

movementTuningData = tuningData[tuningData.action.str[0] == 'm']
movementTuned = movementTuningData.set_index(["animal", "date", "neuron", "action"]).signp.unstack()
fracMovement = movementTuned.any(axis=1).mean(axis=0)
print("Fraction of neurons positively tuned to any of the movements between ports: {:.3f}%".format(fracMovement*100))

#%% SD of example histogram in the tunings figure
histSD = tuningData.query('genotype == "oprm1" & action == "mC2L-"')["tuning"].std()
print("Standard deviation of the tuning scores for mC2L- for all oprm1-neurons: {:.3f}".format(histSD))

#%% Decoding statistics
cachedDataPath = cacheFolder / "decodeWithIncreasingNumberOfNeurons.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisDecoding.decodeWithIncreasingNumberOfNeurons(endoDataPath)
    decodingData.to_pickle(cachedDataPath)
    
accuracyAt200 = decodingData.query("nNeurons == 200").realAccuracy.mean()
print("Average decoding accuracy using 200 neurons: {:.3f}%".format(accuracyAt200*100))

cachedDataPath = cacheFolder / "decodeSortedByMI.pkl"
if cachedDataPath.is_file():
    decodingDataByMI = pd.read_pickle(cachedDataPath)
else:
    decodingDataByMI = analysisDecoding.decodeWithSortedNeurons(endoDataPath)
    decodingDataByMI.to_pickle(cachedDataPath)
descending = decodingDataByMI.query("ordering=='descending'").groupby("nNeurons").accuracy.mean()
ascending = decodingDataByMI.query("ordering=='ascending'").groupby("nNeurons").accuracy.mean()

#%% Decoding the turn
cachedDataPath = cacheFolder / "decodeMovementProgress_mR2C.pkl"
if cachedDataPath.is_file():
    decodingMovementProgress = pd.read_pickle(cachedDataPath)
else:
    decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath)
    decodingMovementProgress.to_pickle(cachedDataPath)
exampleSession = decodingMovementProgress.query("sess == 'oprm1_5308_190131' & not shuffle")

rmsError = np.sqrt(exampleSession.eval("(true - predicted)**2").mean())
avgAbsError = exampleSession.eval("abs(true - predicted)").mean()
print("Average mR2C turn decoding error on session oprm1_5308_190131 is {:.3f}.".format(avgAbsError))
print("\t(The root-mean-square error is {:.3f}.)".format(rmsError))

sess = next(readSessions.findSessions(endoDataPath, animal="5308", date="190131", task="2choice"))
lfa = sess.labelFrameActions(reward="sidePorts")
meanDuration = lfa.query("label == 'mR2C-'").groupby("actionNo").actionDuration.mean().mean()
print("Mean duration of mR2C on session oprm1_5308_190131 is {:.3f} frames ({:.3f}s)".format(meanDuration, meanDuration/20.0))
#%% TODO
# * Average decoding accuracy at 200 neurons
# * Average decoding accuracy weigthed by number of neurons
# * Sorted neurons decoding 100 least informative as accurate as the ? most informative
# * Decoding RME in Fig 4
# * Average right-to-center movement during the example session
