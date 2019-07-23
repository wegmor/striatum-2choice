#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:13:23 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import tqdm
import sklearn.model_selection
import sklearn.svm
import sklearn.metrics
import multiprocessing
import functools
import h5py
import datetime
from utils import readSessions


#%%
def wAvg(group, var, weights):
    return(np.average(group[var], weights=group[weights]))
    
def bootstrap(group, var, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(group[var]), size=len(group[var]),
                               replace=True)
        avgs.append(np.average(group[var].iloc[idx], weights=group[weights].iloc[idx]))
    return(np.std(avgs))
    
def jitter(x, std):
    return(x+np.random.normal(0,std,size=len(x)))
    
    
#%%
def decodeStaySwitchSession(sess, selectedPhase):
    def _prepareTrials(deconv, lfa, selectedPhase):
        avgSig = deconv.groupby(lfa.actionNo).mean()
        labels = lfa.groupby("actionNo").label.first()
        durations = lfa.groupby("actionNo").actionDuration.first() * (1/20)
        
        # append trial type indicators to phase label
        selectedLabels = [selectedPhase+trialType for trialType in ('r.','o!','o.')]
        
        # select valid trials
        validTrials = np.logical_and(avgSig.notna().all(axis=1), labels.isin(selectedLabels))
        durations = durations[validTrials]
        
        # select reward-stay & omission-switch trials (-> SVC data)
        validSVCTrials = np.logical_and(validTrials, labels.str.contains('r.$|o!$'))
        X = avgSig[validSVCTrials]
        Y = labels[validSVCTrials]
        # select valid omission-stay trials (-> to be predicted)
        validOStTrials = np.logical_and(validTrials, labels.str.endswith('o.'))
        Xost = avgSig[validOStTrials]
        Yost = labels[validOStTrials]
        
        return X, Y, Xost, Yost, durations
    
    def _decode(X, Y, Xost, Yost, D):
        splitter = sklearn.model_selection.StratifiedKFold(5, shuffle=True)
        preds = []       # SVC label predictions for reward-stay, omission-switch trials
        dur_preds = []   # SVC predictions based on acction duration not neuronal activity
        P = []           # probabilities for reward-stay, omission-switch trials
        Post = []        # probabilities for omission-stay trials
        C = []           # coefficients
        for i, (train_idx, test_idx) in enumerate(splitter.split(X, Y)):
            # neuronal activity svm
            trainX, trainY = X.iloc[train_idx,:], Y.iloc[train_idx]
            testX = X.iloc[test_idx,:]
            svm = sklearn.svm.SVC(kernel="linear", probability=True,
                                  class_weight='balanced').fit(trainX, trainY)
            
            preds.append(pd.DataFrame(svm.predict(testX), index=testX.index,
                                      columns=['prediction']))
            P.append(pd.DataFrame(svm.predict_proba(testX), index=testX.index,
                                  columns=[c[-2:] for c in svm.classes_]))
            Post.append(pd.DataFrame(svm.predict_proba(Xost), index=Xost.index,
                                     columns=[c[-2:] for c in svm.classes_]))
            C.append(pd.Series(svm.coef_[0]))
            
            # speed svm
            XD = D.loc[X.index].copy() # both use actionNo as index -> get wst-lsw trial durations
            trainD = XD.iloc[train_idx]
            testD = XD.iloc[test_idx]
            dur_svm = (sklearn.svm.SVC(kernel='linear', class_weight='balanced')
                                  .fit(trainD.values.reshape(-1,1), trainY))
            
            dur_preds.append(pd.DataFrame(dur_svm.predict(testD.values.reshape(-1,1)),
                                          index=testD.index,
                                          columns=['prediction']))

        # compute confusion matrix    
        preds = pd.concat(preds).sort_index()
        preds['true'] = Y
        M = sklearn.metrics.confusion_matrix(preds.true, preds.prediction,
                                             labels=svm.classes_)
        M = M / M.sum(axis=1)[:,np.newaxis]
        M = pd.DataFrame(M, index=svm.classes_, columns=svm.classes_)
        M.index.name = 'true'
        M.columns.name = 'predicted'
        M = M.stack().reset_index().rename(columns={0:'percent'})
        M['noNeurons'] = X.shape[1]
        
        # get probabilities for all actions in the selected phase
        P = pd.concat(P)
        # compute average SVM predictions for omission-stay trials
        Post = pd.concat(Post, keys=np.arange(5)).groupby(level=-1).mean()
        P = pd.concat([P,Post]).sort_index()
        P['label'] = pd.concat([Y,Yost])
        P['duration'] = D
        P['prediction'] = preds.prediction # include label predictions for wst-lsw
        dur_preds = pd.concat(dur_preds).sort_index()
        P['duration_prediction'] = dur_preds.prediction # label prediction based on speed
        P = P.reset_index()
        P['noNeurons'] = X.shape[1]
        
        # compute average SVM coefficients
        C = pd.concat(C, keys=np.arange(5)).groupby(level=-1).mean()
        C.index.name = 'neuron'
        C.name = 'coefficient'
        C = pd.DataFrame(C).reset_index()
        
        for k,v in [('date',sess.meta.date),('animal',sess.meta.animal),
                    ('genotype',sess.meta.genotype)]:
            M.insert(0,k,v)
            P.insert(0,k,v)
            C.insert(0,k,v)
        
        return M, P, C
        
    deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
    lfa = sess.labelFrameActions(reward='fullTrial', switch=True)
    if len(deconv) != len(lfa): 
        raise Exception('trace has fewer frames than behavior data!')
    slfa = sess.shuffleFrameLabels(reward='fullTrial', switch=True)[0]
    
    rX, rY, rXost, rYost, rD = _prepareTrials(deconv, lfa, selectedPhase)
    sX, sY, sXost, sYost, sD  = _prepareTrials(deconv, slfa, selectedPhase)
    
    rM, rP, rC = _decode(rX, rY, rXost, rYost, rD)
    sM, sP, sC = _decode(sX, sY, sXost, sYost, sD)
    
    return (rM, rP, rC), (sM, sP, sC)


#%%
def decodeStaySwitch(dataFile, selectedPhase):
    rMs, sMs = pd.DataFrame(), pd.DataFrame()
    rPs, sPs = pd.DataFrame(), pd.DataFrame()
    rCs, sCs = pd.DataFrame(), pd.DataFrame()
    
    for sess in readSessions.findSessions(dataFile, task="2choice"):
        try:
            (rM, rP, rC), (sM, sP, sC) = decodeStaySwitchSession(sess, selectedPhase)
            rMs, sMs = rMs.append(rM, ignore_index=True), sMs.append(sM, ignore_index=True)
            rPs, sPs = rPs.append(rP, ignore_index=True), sPs.append(sP, ignore_index=True)
            rCs, sCs = rCs.append(rC, ignore_index=True), sCs.append(sC, ignore_index=True)
        except:
            continue

    return (rMs, rPs, rCs), (sMs, sPs, sCs)

#%%
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
    
def decodeStaySwitchAcrossDays(dataFile, alignmentFile):
    def _prepareTrials(deconv, lfa, selectedLabels):
        avgSig = deconv.groupby(lfa.actionNo).mean()
        labels = lfa.groupby("actionNo").label.first()
        validTrials = np.logical_and(avgSig.notna().all(axis=1), labels.isin(selectedLabels))
        X = avgSig[validTrials]
        Y = labels[validTrials]
        return X, Y
   
    def _dateDiff(fromDate, toDate):
        fromDate = datetime.datetime.strptime(fromDate, "%y%m%d")
        toDate = datetime.datetime.strptime(toDate, "%y%m%d")
        return (toDate-fromDate).days
    
    alignmentStore = h5py.File(alignmentFile, "r")
    with multiprocessing.Pool(5) as pool:
        acrossDaysResult = []
        for genotype in alignmentStore["data"]:
            for animal in alignmentStore["data/{}".format(genotype)]:
                for fromDate in alignmentStore["data/{}/{}".format(genotype, animal)]:
                    fromSess = next(readSessions.findSessions(dataFile, animal=animal, date=fromDate))
                    fromTask = fromSess.meta.task
                    if fromTask != "2choice": continue
                    fromDeconv = fromSess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
                    fromLfa = fromSess.labelFrameActions(reward="fullTrial", switch=True)
                    if len(fromDeconv) != len(fromLfa): continue
                    suffledLfa = fromSess.shuffleFrameLabels(reward="fullTrial", switch=True)[0]

                    for baseLabel in ("mC2L", "mC2R", "mL2C", "mR2C"):
                        selectedLabels = [baseLabel+"r.", baseLabel+"o!"]
                        fromX, fromY = _prepareTrials(fromDeconv, fromLfa, selectedLabels)
                        shuffledX, shuffledY = _prepareTrials(fromDeconv, suffledLfa, selectedLabels)
                        for toDate in alignmentStore["data/{}/{}/{}".format(genotype, animal, fromDate)]:
                            if toDate <= fromDate: continue
                            match = alignmentStore["data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)][()]

                            toSess = next(readSessions.findSessions(dataFile, animal=animal, date=toDate))
                            toTask = toSess.meta.task
                            if toTask != "2choice": continue
                            toDeconv = toSess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
                            toLfa = toSess.labelFrameActions(reward="fullTrial", switch=True)
                            if len(toDeconv) != len(toLfa): continue

                            if _dateDiff(fromDate, toDate) <= 0: continue
                            toX, toY = _prepareTrials(toDeconv, toLfa, selectedLabels)

                            fcn = functools.partial(_testSameAndNextDay, realX=fromX[match[:,0]], realY=fromY,
                                                    shuffledX=shuffledX[match[:,0]], shuffledY=shuffledY,
                                                    nextX=toX[match[:,1]], nextY=toY)
                            desc = "{} to {} ({})".format(fromSess, toDate, baseLabel)
                            print(desc)
                            for scores in pool.imap(fcn, range(5)):
                                acrossDaysResult.append((genotype, animal, fromDate, toDate,
                                                         fromTask, toTask, baseLabel, match.shape[0],
                                                         fromX.shape[0], toX.shape[0])+scores)
    columns=["genotype", "animal", "fromDate", "toDate", "fromTask",
             "toTask", "label" ,"nNeurons", "nTrialsFrom", "nTrialsTo", "i",
             "sameDayScore",  "nextDayScore", "sameDayShuffled", "nextDayShuffled"]
    return pd.DataFrame(acrossDaysResult, columns=columns)
        