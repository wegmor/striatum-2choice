import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sklearn.svm
import sklearn.ensemble
import sklearn.model_selection
import sklearn.feature_selection
import tqdm
import multiprocessing
import functools
import h5py
import datetime

from utils import readSessions
from utils.cachedDataFrame import cachedDataFrame

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

def _testSameAndNextDay(i, realX, realY, shuffledX, shuffledY, nextX, nextY):
    np.random.seed(np.random.randint(1000000)+i)
    trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(realX, realY,
                                                                            test_size=0.2, stratify=realY)
    svm = sklearn.svm.SVC(kernel="linear").fit(trainX, trainY)
    fromAcc = np.mean(svm.predict(testX) == testY)
    toAcc = np.mean(svm.predict(nextX) == nextY)
    
    trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(shuffledX, shuffledY,
                                                                            test_size=0.2, stratify=shuffledY)
    svm = sklearn.svm.SVC(kernel="linear").fit(trainX, trainY)
    shuffledFromAcc = np.mean(svm.predict(testX) == testY)
    shuffledToAcc   = np.mean(svm.predict(nextX) == nextY)
    
    return (i, fromAcc, toAcc, shuffledFromAcc, shuffledToAcc)

def _dateDiff(fromDate, toDate):
    fromDate = datetime.datetime.strptime(fromDate, "%y%m%d")
    toDate = datetime.datetime.strptime(toDate, "%y%m%d")
    return (toDate-fromDate).days

@cachedDataFrame("decodeWithIncreasingNumberOfNeurons.pkl")
def decodeWithIncreasingNumberOfNeurons(dataFile):
    nShufflesPerNeuronNum = 10
    with multiprocessing.Pool(5) as pool:
        res = []
        for sess in readSessions.findSessions(dataFile, task="2choice"):
            deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
            lfa = sess.labelFrameActions(reward="sidePorts")
            if len(deconv) != len(lfa): continue
            shuffledLfa = sess.shuffleFrameLabels(switch=False)[0]
            realX, realY = _prepareTrials(deconv, lfa)
            shuffledX, shuffledY = _prepareTrials(deconv, shuffledLfa)
            with tqdm.tqdm(total=int(realX.shape[1]/5)*nShufflesPerNeuronNum, desc=str(sess)) as t:
                for nNeurons in range(5, realX.shape[1], 5):
                    fcn = functools.partial(_testRealAndShuffled, realX=realX, realY=realY,
                                            shuffledX=shuffledX, shuffledY=shuffledY, nNeurons=nNeurons)
                    for scores in pool.imap(fcn, range(nShufflesPerNeuronNum)):
                        res.append((str(sess), sess.meta.task, nNeurons)+scores)
                        t.update(1)
    return pd.DataFrame(res, columns=["session", "task", "nNeurons", "i", "realAccuracy", "shuffledAccuracy"])

#def _calcMI(X, Y):
#    mi = list()
#    actionsAsInts = Y.astype("category").cat.codes.values.reshape(-1, 1)
#    for i in range(X.shape[1]):
#        mi.append(sklearn.feature_selection.mutual_info_regression(actionsAsInts,
#                                                                   X[i],
#                                                                   discrete_features=True,
#                                                                   n_neighbors=3)[0])
#    return np.array(mi)


def _launchCrossValScore(i, X, Y):
    np.random.seed(np.random.randint(1000000)+i)
    return i, _crossValScore(X, Y)

@cachedDataFrame("decodeSortedByMI.pkl")
def decodeWithSortedNeurons(dataFile):
    nShufflesPerNeuronNum = 10
    with multiprocessing.Pool(5) as pool:
        res = []
        for sess in readSessions.findSessions(dataFile, task="2choice"):
            deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
            lfa = sess.labelFrameActions(reward="sidePorts")
            if len(deconv) != len(lfa): continue
            X, Y = _prepareTrials(deconv, lfa)
            mutualInformation = sklearn.feature_selection.mutual_info_classif(X, Y)
            ascending = np.argsort(mutualInformation)
            descending = ascending[::-1]
            N = min(201, X.shape[1])
            with tqdm.tqdm(total=int((N-1)/5)*nShufflesPerNeuronNum*2, desc=str(sess)) as t:
                for nNeurons in range(5, N, 5):
                    for ordering in ("ascending", "descending"):
                        selectedNeurons = ascending[:nNeurons] if ordering=="ascending" else descending[:nNeurons]
                        fcn = functools.partial(_launchCrossValScore, X=X[selectedNeurons], Y=Y)
                        for i, score in pool.imap(fcn, range(nShufflesPerNeuronNum)):
                            res.append((str(sess), sess.meta.task, nNeurons, i, ordering, score))
                            t.update(1)
    return pd.DataFrame(res, columns=["session", "task", "nNeurons", "i", "ordering", "accuracy"])

@cachedDataFrame("decodeConfusion.pkl")
def decodingConfusion(dataFile):
    confMats = []
    for sess in readSessions.findSessions(dataFile, task="2choice"):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        lfa = sess.labelFrameActions(reward="sidePorts")
        if len(deconv) != len(lfa): continue
        realX, realY = _prepareTrials(deconv, lfa)
        for i in tqdm.trange(5, desc=str(sess)):
            trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(realX, realY,
                                                                                test_size=0.2, stratify=realY)
            svm = sklearn.svm.SVC(kernel="linear").fit(trainX, trainY)
            pred = svm.predict(testX)
            m = sklearn.metrics.confusion_matrix(testY, pred)
            m = pd.DataFrame(m, index=svm.classes_, columns=svm.classes_)
            m = m.rename_axis(index="true", columns="predicted").unstack()
            m = m.rename("occurences").reset_index()
            m["sess"] = str(sess)
            m["i"] = i
            m["nNeurons"] = deconv.shape[1]
            confMats.append(m)
    return pd.concat(confMats)

@cachedDataFrame("decodingAcrossDays.pkl")
def decodingAcrossDays(dataFile, alignmentFile):
    alignmentStore = h5py.File(alignmentFile, "r")
    with multiprocessing.Pool(5) as pool:
        acrossDaysResult = []
        for genotype in alignmentStore["data"]:
            for animal in alignmentStore["data/{}".format(genotype)]:
                for fromDate in alignmentStore["data/{}/{}".format(genotype, animal)]:
                    fromSess = next(readSessions.findSessions(dataFile, animal=animal, date=fromDate))
                    fromTask = fromSess.meta.task
                    if fromTask == "openField": continue
                    fromDeconv = fromSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
                    fromLfa = fromSess.labelFrameActions(reward="sidePorts")
                    if len(fromDeconv) != len(fromLfa): continue
                    suffledLfa = fromSess.shuffleFrameLabels(switch=False)[0]
                    fromX, fromY = _prepareTrials(fromDeconv, fromLfa)
                    shuffledX, shuffledY = _prepareTrials(fromDeconv, suffledLfa)
                    for toDate in alignmentStore["data/{}/{}/{}".format(genotype, animal, fromDate)]:
                        if toDate <= fromDate: continue
                        match = alignmentStore["data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)][()]

                        toSess = next(readSessions.findSessions(dataFile, animal=animal, date=toDate))
                        toTask = toSess.meta.task
                        if toTask == "openField": continue
                        toDeconv = toSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
                        toLfa = toSess.labelFrameActions(reward="sidePorts")
                        if len(toDeconv) != len(toLfa): continue

                        if _dateDiff(fromDate, toDate) <= 0: continue
                        toX, toY = _prepareTrials(toDeconv, toLfa)

                        fcn = functools.partial(_testSameAndNextDay, realX=fromX[match[:,0]], realY=fromY,
                                                shuffledX=shuffledX[match[:,0]], shuffledY=shuffledY,
                                                nextX=toX[match[:,1]], nextY=toY)
                        for scores in tqdm.tqdm(pool.imap(fcn, range(5)), total=5, desc="{} to {}".format(fromSess, toDate)):
                            acrossDaysResult.append((genotype, animal, fromDate, toDate,
                                                     fromTask, toTask, match.shape[0])+scores)
    columns=["genotype", "animal", "fromDate", "toDate", "fromTask",
             "toTask", "nNeurons", "i", "sameDayScore",  "nextDayScore",
             "sameDayShuffled", "nextDayShuffled"]
    return pd.DataFrame(acrossDaysResult, columns=columns)


def decodeMovementProgress(dataFile, label="mR2C-"):
    @cachedDataFrame("decodeMovementProgress_{}.pkl".format(label[:4]))
    def cachedVersion():
        return _decodeMovementProgress(dataFile, label)
    return cachedVersion()

def _decodeMovementProgress(dataFile, label):
    allSess = []
    for sess in readSessions.findSessions(dataFile, task="2choice"):
        for shuffle in (False, True):
            if shuffle:
                lfa = sess.shuffleFrameLabels(switch=False)[0]
            else:
                lfa = sess.labelFrameActions(reward="sidePorts")
            deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
            if len(lfa) != len(deconv): continue
            if deconv.isna().any().any(): continue #TODO: Fix this
            X = deconv[lfa.label==label]
            Y = lfa.actionProgress[lfa.label==label]

            actionNos = lfa.actionNo[lfa.label==label]

            XactionNo = X.set_index(actionNos).sort_index()
            YactionNo = pd.Series(Y.values, index=XactionNo.index)

            splitter = sklearn.model_selection.KFold(5, shuffle=True)
            uniqueActionNos = actionNos.unique()

            for trainInd, testInd in tqdm.tqdm(splitter.split(uniqueActionNos), total=5, desc=str(sess)):
                trainActionNos = uniqueActionNos[trainInd]
                testActionNos = uniqueActionNos[testInd]
                trainX = XactionNo.loc[trainActionNos]
                trainY = YactionNo.loc[trainActionNos]
                testX = XactionNo.loc[testActionNos]
                testY = YactionNo.loc[testActionNos]

                classifier = sklearn.linear_model.LinearRegression()
                classifier.fit(trainX, trainY)
                pred = classifier.predict(testX)
                allSess.append(pd.DataFrame({'true': testY, 'predicted': pred, 'sess': str(sess),
                                             'nNeurons': X.shape[1], 'nTrials': len(uniqueActionNos),
                                             'shuffle': shuffle, 'label': label}))
    return pd.concat(allSess)

def get_centers(rois):
    # find pixel of maximum intensity in each mask; use as neuron center
    centers = np.array(np.unravel_index(np.array([np.argmax(roi) for roi in rois]),
                                                  rois.shape[1:]))
    centers = centers[::-1].T
    return(centers)