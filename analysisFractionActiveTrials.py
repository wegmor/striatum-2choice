import numpy as np
import pandas as pd
import tqdm
from utils import readSessions
from utils.cachedDataFrame import cachedDataFrame

@cachedDataFrame("fractionActiveTrials.pkl")
def getFractionActiveTrials(dataFile, threshold=0.5):
    dfs = []
    for sess in tqdm.tqdm(readSessions.findSessions(dataFile, task="2choice"), total=66):
        traces = sess.readDeconvolvedTraces(zScore=True)
        lfa = sess.labelFrameActions(reward="sidePorts", switch=False)
        if len(traces) != len(lfa): continue
        lfa = lfa.set_index(traces.index)

        meanActivities = traces.groupby(lfa.actionNo).mean()
        activeTrials = (meanActivities>threshold)
        lfaLabels = lfa.groupby("actionNo").label.first()
        fracTrialsActive = activeTrials.groupby(lfaLabels).mean()
        fracTrialsActive = fracTrialsActive.rename_axis(columns="neuron").stack().rename("fracTrialsActive")
        avgActivity = meanActivities[activeTrials].groupby(lfaLabels).mean()
        avgActivity = avgActivity.rename_axis(columns="neuron").stack().rename("avgActivity")
        df = pd.concat((fracTrialsActive, avgActivity), axis=1).reset_index()
        df.insert(0, "sess", str(sess))
        dfs.append(df)
    return pd.concat(dfs)