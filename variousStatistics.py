import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import h5py
import pathlib

from utils import readSessions
import analysisTunings

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
