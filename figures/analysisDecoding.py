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

import readSessions

#The bins that the decoder needs to distinguish
selectedLabels = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pC2L-", "pC2R-",
                  "pL2Cd", "pL2Co", "pL2Cr", "pR2Cd", "pR2Co", "pR2Cr"]

def _crossValScore(X, Y):
    svm = sklearn.svm.SVC(kernel="linear", cache_size=2000)
    trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, stratify=Y)
    svm.fit(trainX, trainY)
    predicted = svm.predict(testX)
    accuracy = np.mean(predicted == testY)
    return accuracy

def _testRealAndShuffled(i, realX, realY, shuffledX, shuffledY, nNeurons):
    np.random.seed(np.random.randint(1000000)+i) #Seed each process differently
    neurons = np.random.choice(realX.shape[1], nNeurons, replace=False)
    realScore = _crossValScore(realX[neurons], realY)
    shuffledScore = _crossValScore(shuffledX[neurons], shuffledY)
    return (i, realScore, shuffledScore)

def _prepareTrials(deconv, lfa):
    avgSig = deconv.groupby(lfa.actionNo).mean()
    labels = lfa.groupby("actionNo").label.first()
    validTrials = np.logical_and(avgSig.notna().all(axis=1), labels.isin(selectedLabels))
    X = avgSig[validTrials]
    Y = labels[validTrials]
    return X, Y

def decodeWithIncreasingNumberOfNeurons(dataFile):
    nShufflesPerNeuronNum = 10
    with multiprocessing.Pool(5) as pool:
        res = []
        for sess in readSessions.findSessions(dataFile, task="2choice"):
            deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
            lfa = sess.labelFrameActions(reward="sidePorts")
            if len(deconv) != len(lfa): continue
            suffledLfa = sess.shuffleFrameLabels(switch=False)[0]
            realX, realY = _prepareTrials(deconv, lfa)
            shuffledX, shuffledY = _prepareTrials(deconv, suffledLfa)
            with tqdm.tqdm(total=int(realX.shape[1]/5)*nShufflesPerNeuronNum, desc=str(sess)) as t:
                for nNeurons in range(5, realX.shape[1], 5):
                    fcn = functools.partial(_testRealAndShuffled, realX=realX, realY=realY,
                                            shuffledX=shuffledX, shuffledY=shuffledY, nNeurons=nNeurons)
                    for scores in pool.imap(fcn, range(nShufflesPerNeuronNum)):
                        res.append((str(sess), sess.meta.task, nNeurons)+scores)
                        t.update(1)
    return pd.DataFrame(res, columns=["session", "task", "nNeurons", "i", "realAccuracy", "shuffledAccuracy"])