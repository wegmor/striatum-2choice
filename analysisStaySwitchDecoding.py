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


#%%
def _decodeStaySwitchSession(sess, selectedPhase):
    def decode(X, Y, Xost, Yost, D):
        splitter = sklearn.model_selection.StratifiedKFold(5, shuffle=True)
        preds = []  # SVC label predictions for reward-stay, omission-switch trials
        P = []      # probabilities for reward-stay, omission-switch trials
        Post = []   # probabilities for omission-stay trials
        C = []      # coefficients
        for i, (train_idx, test_idx) in enumerate(splitter.split(X, Y)):
            trainX, trainY = X.iloc[train_idx,:], Y.iloc[train_idx]
            testX = X.iloc[test_idx,:]
            svm = sklearn.svm.SVC(kernel="linear", probability=True,
                                  class_weight='balanced').fit(trainX, trainY)
            
            preds.append(pd.DataFrame(svm.predict(testX), index=testX.index,
                                      columns=['prediction']))
            P.append(pd.DataFrame(svm.predict_proba(testX), index=testX.index,
                                  columns=svm.classes_))
            Post.append(pd.DataFrame(svm.predict_proba(Xost), index=Xost.index,
                                     columns=svm.classes_))
            C.append(pd.Series(svm.coef_[0]))
        
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
    
    rM, rP, rC = decode(rX, rY, rXost, rYost, rD)
    sM, sP, sC = decode(sX, sY, sXost, sYost, sD)
    
    return (rM, rP, rC), (sM, sP, sC)


#%%
def decodeStaySwitch(dataFile, selectedPhase):
    rMs, sMs = pd.DataFrame(), pd.DataFrame()
    rPs, sPs = pd.DataFrame(), pd.DataFrame()
    rCs, sCs = pd.DataFrame(), pd.DataFrame()
    
    for sess in readSessions.findSessions(dataFile, task="2choice"):
        try:
            (rM, rP, rC), (sM, sP, sC) = _decodeStaySwitchSession(sess, selectedPhase)
            rMs, sMs = rMs.append(rM, ignore_index=True), sMs.append(sM, ignore_index=True)
            rPs, sPs = rPs.append(rP, ignore_index=True), sPs.append(sP, ignore_index=True)
            rCs, sCs = rCs.append(rC, ignore_index=True), sCs.append(sC, ignore_index=True)
        except:
            continue

    return (rMs, rPs, rCs), (sMs, sPs, sCs)

