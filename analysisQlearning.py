import numpy as np
import pandas as pd
import tqdm
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyximport; pyximport.install()
from utils import readSessions, qActionValues
from utils.cachedDataFrame import cachedDataFrame
import style

@cachedDataFrame("qParameters.pkl")
def fitQParameters(datafile):
    params = []
    for sess in tqdm.tqdm(readSessions.findSessions(datafile, task="2choice"),
                          desc="Fitting Q-parameters"):
        lfa = sess.labelFrameActions(reward="fullTrial", switch=True)
        perAction = lfa.groupby("actionNo").first()
        perAction = perAction[perAction.label.str.match("p[RL]2C")]
        rightPort = (perAction.label.str[1] == 'R').values.astype(np.int8)
        rewarded = (perAction.label.str[-2] == 'r').values.astype(np.int8)
        L = lambda x: -qActionValues.qLikelihood(rightPort, rewarded, x[0], x[1], x[2])
        params.append((str(sess),)+tuple(scipy.optimize.fmin(L, np.array((0.1, 1.0, 0.0)), disp=0)))
    return pd.DataFrame(params, columns=["session", "alpha", "beta", "bias"]).set_index("session")

@cachedDataFrame("qParametersPerAnimal.pkl")
def fitQParametersPerAnimal(datafile):
    animals = pd.read_hdf(datafile, "meta").animal.unique()
    params = []
    for animal in tqdm.tqdm(animals, desc="Fitting Q-parameters"):
        data = []
        for sess in readSessions.findSessions(datafile, animal=animal, task="2choice",
                                              onlyRecordedTrials=False):
            sv = sess.readSensorValues(onlyRecording=False, reindexFrameNo=False)
            lfa = sess.labelFrameActions(sv, reward="fullTrial", switch=True)
            perAction = lfa.groupby("actionNo").first()
            perAction = perAction[perAction.label.str.match("p[RL]2C")]
            rightPort = (perAction.label.str[1] == 'R').values.astype(np.int8)
            rewarded = (perAction.label.str[-2] == 'r').values.astype(np.int8)
            data.append((rightPort, rewarded))
        def L(x):
            res = 0
            for rightPort, rewarded in data:
                res -= qActionValues.qLikelihood(rightPort, rewarded, x[0], x[1], x[2])
            return res
        params.append((animal,)+tuple(scipy.optimize.fmin(L, np.array((0.1, 1.0, 0.0)), disp=0)))
    return pd.DataFrame(params, columns=["animal", "alpha", "beta", "bias"]).set_index("animal")

@cachedDataFrame("qActionValues.pkl")
def getQActionValues(datafile, perAnimal=False):
    if perAnimal:
        params = fitQParametersPerAnimal(datafile)
    else:
        params = fitQParameters(datafile)
    res = []
    for sess in tqdm.tqdm(readSessions.findSessions(datafile, task="2choice"),
                          desc="Calculating Q-action values"):
        lfa = sess.labelFrameActions(reward="fullTrial", switch=True)
        perAction = lfa.groupby("actionNo").first()
        outcomeActions = perAction[perAction.label.str.match("p[RL]2C")].copy()
        rightPort = (outcomeActions.label.str[1] == 'R').values.astype(np.int8)
        rewarded = (outcomeActions.label.str[-2] == 'r').values.astype(np.int8)
        key = sess.meta.animal if perAnimal else str(sess)
        alpha = params.loc[key].alpha
        beta = params.loc[key].beta
        bias = params.loc[key].bias
        outcomeActions["Qr_minus_Ql"] = qActionValues.qActionValues(rightPort, rewarded, alpha)
        outcomeActions["Q_actionValue"] = beta*outcomeActions["Qr_minus_Ql"] + bias
        outcomeActions["P_r"] = 1/(1+np.exp(-outcomeActions["Q_actionValue"]))
        df = outcomeActions[["Qr_minus_Ql", "Q_actionValue", "P_r"]]
        df = perAction[["label"]].join(df, on="actionNo")
        df["session"] = str(sess)
        df = df.bfill()
        res.append(df.reset_index().set_index(["session", "actionNo"]))
    return pd.concat(res)
