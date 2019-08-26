import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sklearn.svm
import sklearn.ensemble
import sklearn.model_selection
import tqdm
import multiprocessing
import functools
import pyximport; pyximport.install()

from striatum_2choice.utils import readSessions
from striatum_2choice.utils import particleFilter
from striatum_2choice.utils import segmentBehaviors

def segmentAllOpenField(datafile):
    allBehaviors = []
    for sess in readSessions.findSessions(datafile, task="openField"):
        tracking = sess.readTracking(inCm=True)
        coords = particleFilter.particleFilter(tracking)
        coords.rename(columns={"bodyAngle": "bodyDirection"}, inplace=True)
        coords.rename_axis("time", axis=0, inplace=True)
        coords.bodyDirection *= 180/np.pi
        behaviors = segmentBehaviors.segmentBehaviors(coords)
        behaviors.insert(0, "session", str(sess))
        behaviors.insert(1, "actionNo", behaviors.index.copy())
        allBehaviors.append(behaviors)
    return pd.concat(allBehaviors)
