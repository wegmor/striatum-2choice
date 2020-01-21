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
from itertools import product
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import readSessions, fancyViz
import statsmodels.api as sm
import style
import sklearn.metrics

style.set_context()
plt.ioff()


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
    print(str(sess)+' '+selectedPhase)
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
        validSVCTrials = np.logical_and(validTrials, labels.str.contains('r\.$|o!$'))
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
        
    deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
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
        #print([(l,(Y == l).sum()) for l in Y.unique()])
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
                    fromDeconv = fromSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
                    fromLfa = fromSess.labelFrameActions(reward="fullTrial", switch=True)
                    if len(fromDeconv) != len(fromLfa): continue
                    suffledLfa = fromSess.shuffleFrameLabels(reward="fullTrial", switch=True)[0]

                    for baseLabel in ("pC2L", "mC2L", "pC2R", "mC2R", "mL2C", "mR2C", "dL2C", "dR2C"):
                        selectedLabels = [baseLabel+"r.", baseLabel+"o!"]
                        fromX, fromY = _prepareTrials(fromDeconv, fromLfa, selectedLabels)
                        shuffledX, shuffledY = _prepareTrials(fromDeconv, suffledLfa, selectedLabels)
                        for toDate in alignmentStore["data/{}/{}/{}".format(genotype, animal, fromDate)]:
                            if toDate <= fromDate: continue
                            match = alignmentStore["data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)][()]

                            toSess = next(readSessions.findSessions(dataFile, animal=animal, date=toDate))
                            toTask = toSess.meta.task
                            if toTask != "2choice": continue
                            toDeconv = toSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
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


#%%
def predictStaySwitchAcrossDays(dataFile, alignmentFile):
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
    
    def _predictNextDay(realX, realY, nextX, nextY):
        svm = sklearn.svm.SVC(kernel="linear", probability=True,
                              class_weight='balanced').fit(realX, realY)
        probabilities = pd.DataFrame(svm.predict_proba(nextX), index=nextX.index,
                                     columns=[c[-2:] for c in svm.classes_])
        probabilities['label'] = nextY
        return probabilities.reset_index()
    
    alignmentStore = h5py.File(alignmentFile, "r")
    P = pd.DataFrame()

    # load sessions meta data, reduce to imaged sessions of animals that performed FA
    meta = pd.read_hdf(dataFile, 'meta')
    meta = meta.loc[(meta.caRecordings.str.len() != 0) &
                    (meta.task.isin(['2choice','forcedAlternation','2choiceAgain'])) &
                    (meta.animal.isin(meta.query('task == "forcedAlternation"').animal.unique()))
                   ].copy()
    # sort index by date
    meta['date_fmt'] = pd.to_datetime(meta.date, yearfirst=True)
    meta = meta.set_index(['genotype','animal','date_fmt']).sort_index()
    # count recording sessions backwards from last -1 to first
    meta['noRecSessions'] = meta.groupby(['genotype','animal']).size()
    meta['recSession'] = np.concatenate([np.arange(-n,0) for n in 
                                         meta.groupby(['genotype','animal']).noRecSessions.first()])
    meta = meta.reset_index().set_index(['genotype','animal','date'])['recSession']
        
    for genotype in alignmentStore["data"]:
        for animal in alignmentStore["data/{}".format(genotype)]:
            if animal not in meta.reset_index().animal.unique(): continue
            for toDate in alignmentStore["data/{}/{}".format(genotype, animal)]:
                toSess = next(readSessions.findSessions(dataFile, animal=animal, date=toDate))
                toTask = toSess.meta.task
                if toTask not in ["forcedAlternation","2choiceAgain"]: continue
                if meta.loc[genotype, animal, toDate] not in [-4,-3,-2,-1]: continue
                toDeconv = toSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
                toLfa = toSess.labelFrameActions(reward="fullTrial", switch=True)
                if len(toDeconv) != len(toLfa): continue

                for baseLabel in ("pC2L","pC2R","mC2L","mC2R","dL2C","dR2C","mL2C","mR2C"):
                    selectedLabels = [baseLabel+"r.", baseLabel+"r!", baseLabel+"o.", baseLabel+"o!"]
                    toX, toY = _prepareTrials(toDeconv, toLfa, selectedLabels)

                    for fromDate in alignmentStore["data/{}/{}/{}".format(genotype, animal, toDate)]:
                        match = alignmentStore["data/{}/{}/{}/{}/match".format(genotype, animal,
                                                                               fromDate, toDate)][()]

                        fromSess = next(readSessions.findSessions(dataFile, animal=animal, date=fromDate))
                        fromTask = fromSess.meta.task
                        if fromTask != "2choice": continue
                        if meta.loc[genotype, animal, fromDate] not in [-7,-6,-5]: continue
                        fromDeconv = fromSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
                        fromLfa = fromSess.labelFrameActions(reward="fullTrial", switch=True)
                        if len(fromDeconv) != len(fromLfa): continue

                        #if _dateDiff(fromDate, toDate) <= 7: continue
                        selectedLabels = [baseLabel+"r.", baseLabel+"o!"]
                        fromX, fromY = _prepareTrials(fromDeconv, fromLfa, selectedLabels)

                        desc = "{} to {} ({})".format(fromSess, toDate, baseLabel)
                        print(desc)
                        probabilities = _predictNextDay(realX=fromX[match[:,0]], realY=fromY,
                                                        nextX=toX[match[:,1]], nextY=toY)
                        for k,v in [('fromDate', fromDate),
                                    ('toDate', toDate),
                                    ('fromTask', fromTask),
                                    ('toTask', toTask),
                                    ('fromRecSession', meta.loc[genotype, animal, fromDate]),
                                    ('toRecSession', meta.loc[genotype, animal, toDate]),
                                    ('animal', animal),
                                    ('genotype', genotype),
                                    ('noNeurons', match.shape[0])]:
                            probabilities.insert(0, k, v)
                        P = P.append(probabilities, ignore_index=True)
    return P
      

#%%
def crossDecodeStaySwitch(dataFile):
    def _prepareTrials(deconv, lfa, selectedLabels): # TODO: exact copy of above
        avgSig = deconv.groupby(lfa.actionNo).mean()
        labels = lfa.groupby("actionNo").label.first()
        validTrials = np.logical_and(avgSig.notna().all(axis=1), labels.isin(selectedLabels))
        X = avgSig[validTrials]
        Y = labels[validTrials]
        return X, Y
    
    def _decodeSession(Xs, Ys):
        results = pd.DataFrame()
        for trainAction, trainY in Ys.groupby(Ys.str.slice(0,4)):
            trainX = Xs.loc[trainY.index] # index by actionNo
            svm = (sklearn.svm.SVC(kernel="linear", class_weight='balanced')
                              .fit(trainX, trainY))
            
            testYs = Ys[Ys.str.slice(0,4) != trainAction]
            for testAction, testY in testYs.groupby(testYs.str.slice(0,4)):
                testX = Xs.loc[testY.index]
                pred = pd.Series(svm.predict(testX), index=testX.index)
                accuracy = np.mean(pred.str.slice(-2) == testY.str.slice(-2))
                results = results.append({'genotype': sess.meta.genotype,
                                          'animal': sess.meta.animal,
                                          'date': sess.meta.date,
                                          'noNeurons': deconv.shape[1],
                                          'trainAction': trainAction,
                                          'testAction': testAction,
                                          'accuracy': accuracy},
                                         ignore_index=True)
        return results    
        
    shuffledCrossDecode = pd.DataFrame()
    realCrossDecode = pd.DataFrame()
    for sess in readSessions.findSessions(dataFile, task='2choice'):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        lfa = sess.labelFrameActions(reward='fullTrial', switch=True)
        if len(deconv) != len(lfa): continue
        slfa = sess.shuffleFrameLabels(reward='fullTrial', switch=True)[0]
        selectedLabels = [base+trial for base in ['dL2C','mL2C','pC2L','mC2L',
                                                  'dR2C','mR2C','pC2R','mC2R']
                                     for trial in ['r.','o!']]
    
        rXs, rYs = _prepareTrials(deconv, lfa, selectedLabels)
        sXs, sYs = _prepareTrials(deconv, slfa, selectedLabels)
        
        realCrossDecode = realCrossDecode.append(_decodeSession(rXs, rYs),
                                                 ignore_index=True)
        shuffledCrossDecode = shuffledCrossDecode.append(_decodeSession(sXs, sYs),
                                                         ignore_index=True)
    return realCrossDecode, shuffledCrossDecode


#%%
def getRegressionVars(sensorValues, trials_back=7):
    sv = sensorValues.copy()
    
    # left exit -> beamL decrements from 1 to 0, etc.
    sv['leftEx'] = sv.beamL.diff() == -1
    sv['rightEx'] = sv.beamR.diff() == -1
    sv['leftIn'] = sv.beamL.diff() == 1
    sv['rightIn'] = sv.beamR.diff() == 1
    
    # no of choice port exits
    sv['sideExNo'] = np.cumsum(sv.leftEx | sv.rightEx)
    
    # reduce df to choice port exits
    df = sv.loc[sv.leftEx | sv.rightEx,
                ['leftEx','rightEx','sideExNo','rewardNo']].copy()
    
    # define reward -- it is delivered when the beam is still broken,
    # after 350 ms delay, before port exit
    df['reward'] = (df.rewardNo.diff() >= 1).astype('bool')
    
    # switch: next exit is from a different port than current
    df['switch'] = df.leftEx.astype('int').diff().abs()
    
    # convert to int
    df = df.dropna()
    df['leftEx'] = df.leftEx.astype('int')
    df['rightEx'] = df.rightEx.astype('int')
    
    # get Y & N (Y=1 if left and rewarded, -1 if right and rewarded)
    df['Y0'] = df.reward * (df.leftEx - df.rightEx)
    df['N0'] = ~df.reward * (df.leftEx - df.rightEx) # ~ requires bool!


    df['intercept'] = 1.0
    reg_vars = ['intercept']
    # get shifts
    for j in range(1,trials_back+1):
        df['Y{}'.format(j)] = df.Y0.shift(j)
        df['N{}'.format(j)] = df.N0.shift(j)
        reg_vars += ['Y{}'.format(j), 'N{}'.format(j)]
    
    # Y1 should be outcome of last trial, i.e. before leaving side-port last;
    # without v, Y0 is that outcome
    df['sideExNo'] -= 1
    # v drops index!
    sv = sv.merge(df[['sideExNo','switch',*reg_vars]], how='left', on='sideExNo')
    return sv[['leftIn','rightIn','switch',*reg_vars]].copy(), reg_vars
    

def getAVCoefficients(dataFile):
    regression_df = pd.DataFrame()
    for sess in readSessions.findSessions(dataFile, task='2choice',
                                          onlyRecordedTrials=False):
        sv = sess.readSensorValues(onlyRecording=False) 
        df, reg_vars = getRegressionVars(sv)
        df = df.loc[df.leftIn | df.rightIn].copy()
        df['genotype'] = sess.meta.genotype
        df['animal'] = sess.meta.animal
        regression_df = regression_df.append(df.dropna())
    
    coefficients =  pd.DataFrame() # stores coefficient for each animal
    regression_df = regression_df.set_index(['genotype','animal']).sort_index()
    # loop through animals and run regressions
    for (genotype, animal), df in regression_df.groupby(['genotype','animal']):
        df = df.copy()
        df['intercept'] = 1.0         
        logit = sm.Logit(df.leftIn, df[reg_vars])
        result = logit.fit(use_t=True, disp=False)
        
        coef = result.params
        coef['genotype'] = genotype
        coef['animal'] = animal
        coefficients = coefficients.append(coef, ignore_index=True)
        regression_df.loc[(genotype,animal), 'prediction'] = result.predict(df[reg_vars])
        
    coefficients = coefficients.set_index(['genotype','animal']).sort_index()
    regression_df['value'] = ((regression_df[reg_vars] * coefficients)
                                  .sum(axis=1, skipna=False).values)
    regression_df = regression_df[['leftIn','rightIn','switch','prediction','value']]

    return coefficients, regression_df


def getActionValues(dataFile, coefficients, on_shuffled=False):
    def lfa2sv(lfa):
        lfa['beamL'] = lfa.label.str.contains('[dp]L').astype('int')
        lfa['beamR'] = lfa.label.str.contains('[dp]R').astype('int')
        lfa['rewardNo'] = (lfa.label.str.contains('p[LR]2.r[\.!]$')
                              .astype('int').diff() == 1).cumsum()
        return lfa[['beamL','beamR','rewardNo']]
    
    actionValues = pd.DataFrame()   
    for sess in readSessions.findSessions(dataFile, task='2choice'):
        if on_shuffled:
            lfa = sess.shuffleFrameLabels(reward='fullTrial', switch=True)[0]
            sv = lfa2sv(lfa)
        else:
            lfa = sess.labelFrameActions(reward='fullTrial', switch=True)
            sv = sess.readSensorValues()
        dummy_df, reg_vars = getRegressionVars(sv) # cf above; merge resets index in dummy_df
        lfa['value'] = (dummy_df[reg_vars] * \
                        coefficients.loc[sess.meta.genotype, sess.meta.animal]
                       ).sum(axis=1, skipna=False)
        lfa = lfa.groupby('actionNo').first().reset_index()
        
        for k,v in [('date',sess.meta.date), ('animal',sess.meta.animal),
                    ('genotype',sess.meta.genotype)]:
            lfa.insert(0,k,v)
        
        
        actionValues = actionValues.append(lfa[['genotype','animal','date',
                                                'actionNo','label','value']],
                                           ignore_index=True)
    return actionValues


#%%
#def getWStayLSwitchAUC(dataFile, n_shuffles=1000, on_shuffled=False):
#    def _getAUC(labels, avgs):
#        fpr, tpr, _ = roc_curve(labels, avgs, pos_label='r.')
#        roc_auc = 2*(auc(fpr, tpr)-.5) # Gini coefficient!
#        return roc_auc
#        
#    auc_df = pd.DataFrame()
#    for sess in readSessions.findSessions(dataFile, task='2choice'):
#        print(str(sess))
#        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
#        if on_shuffled:
#            lfa = sess.shuffleFrameLabels(reward='fullTrial', switch=True)[0]
#        else:
#            lfa = sess.labelFrameActions(reward='fullTrial', switch=True)
#        if len(deconv) != len(lfa): continue
#        trialAvgs = deconv.groupby(lfa.actionNo).mean() # trial-average
#        labels = lfa.groupby("actionNo").label.first()
#        selectedLabels = [base+trial for base in ['dL2C','pL2C','mL2C','pC2L','mC2L',
#                                                  'dR2C','pR2C','mR2C','pC2R','mC2R']
#                                     for trial in ['r.','o!']]
#        validTrials = np.logical_and(trialAvgs.notna().all(axis=1), labels.isin(selectedLabels))
#        labels = pd.DataFrame(labels.loc[validTrials])
#        trialAvgs = trialAvgs.loc[validTrials]
#        
#        # v nasty hack
#        labels['label'] = labels.label.replace({'pC2Lo!':'pC2Ro!', 'pC2Ro!':'pC2Lo!'})
#        labels['action'], labels['trialType'] = (labels.label.str.slice(0,4),
#                                                 labels.label.str.slice(4))
#        
#        df = pd.DataFrame()
#        for action, ls in labels.groupby('action'):
#            lsAvgs = trialAvgs.loc[ls.index].copy()
#            for n in lsAvgs: # iterate over neurons
#                roc_auc = _getAUC(ls.trialType.values, lsAvgs[n].values)
#                # v AUC for shuffled r./o! labels
#                shuffle_dist = [_getAUC(np.random.permutation(ls.trialType.values),
#                                        lsAvgs[n].values) for _ in range(n_shuffles)]
#                shuffle_dist = np.array(shuffle_dist)
#                pct = np.searchsorted(np.sort(shuffle_dist), roc_auc) / len(shuffle_dist)
#                df = df.append(pd.Series({'neuron':n, 'auc':roc_auc, 'pct':pct,
#                                          'action':action,
#                                          's_mean':shuffle_dist.mean(),
#                                          's_std':shuffle_dist.std(),
#                                          'tuning':(roc_auc-shuffle_dist.mean())/shuffle_dist.std()}),
#                               ignore_index=True)
#        for k,v in [('date',sess.meta.date), ('animal',sess.meta.animal), 
#                    ('genotype',sess.meta.genotype)]:
#            df.insert(0,k,v)
#        
#        auc_df = auc_df.append(df, ignore_index=True)
#    return auc_df


#%%
def getWStayLSwitchAUC(dataFile, n_shuffles=1000, on_shuffled=False): # shit is slower than hell
    def _prepareTrials(deconv, lfa):
        trialAvgs = deconv.groupby(lfa.actionNo).mean() # trial-average
        labels = lfa.groupby("actionNo").label.first()
        selectedLabels = [base+trial for base in ['dL2C','pL2C','mL2C','pC2L','mC2L',
                                                  'dR2C','pR2C','mR2C','pC2R','mC2R']
                                     for trial in ['r.','o!']]
        validTrials = np.logical_and(trialAvgs.notna().all(axis=1), labels.isin(selectedLabels))
        labels = pd.DataFrame(labels.loc[validTrials])
        trialAvgs = trialAvgs.loc[validTrials]
        # v nasty hack (use same origin for switch trials)
        #labels['label'] = labels.label.replace({'pC2Lo!':'pC2Ro!', 'pC2Ro!':'pC2Lo!'})
        labels['action'], labels['trialType'] = (labels.label.str.slice(0,4),
                                                 labels.label.str.slice(4))
        return trialAvgs, labels
    
    def _getAUCs(avgs, labels):
        def _getAUC(labels, avgs):
            fpr, tpr, _ = sklearn.metrics.roc_curve(labels, avgs, pos_label='r.')
            roc_auc = 2*(sklearn.metrics.auc(fpr, tpr)-.5) # Gini coefficient!
            return roc_auc
        df = pd.DataFrame()
        for action, ls in labels.groupby('action'): # iterate over actions
            lsAvgs = avgs.loc[ls.index].copy()
            for n in lsAvgs: # iterate over neurons
                roc_auc = _getAUC(ls.trialType.values, lsAvgs[n].values)
                df = df.append(pd.Series({'neuron':n, 'auc':roc_auc,
                                          'action':action}),
                               ignore_index=True)
        return df
    
    def _getPctl(dist, value):
        return np.searchsorted(np.sort(dist), value) / len(dist)
        
    auc_df = pd.DataFrame()
    for sess in readSessions.findSessions(dataFile, task='2choice'):
        print(str(sess))
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        if on_shuffled:
            lfa = sess.shuffleFrameLabels(reward='fullTrial', switch=True)[0]
        else:
            lfa = sess.labelFrameActions(reward='fullTrial', switch=True)
        if len(deconv) != len(lfa): continue
    
        trialAvgs, labels = _prepareTrials(deconv, lfa)
        shuffled_data = [_prepareTrials(deconv, slfa) 
                         for slfa in sess.shuffleFrameLabels(reward='fullTrial',
                                                             switch=True,
                                                             n=n_shuffles)]
                
        realAUCs = _getAUCs(trialAvgs, labels).set_index(['action','neuron'])
        shuffledAUCs = pd.concat([_getAUCs(*sdata) for sdata in shuffled_data],
                                 keys=np.arange(len(shuffled_data)), names=['shuffleNo'])
        shuffledAUCs = (shuffledAUCs.reset_index(-1, drop=True)
                                    .set_index(['action','neuron'], append=True)
                                    .unstack('shuffleNo')['auc'])
        
        realAUCs['pct'] = np.nan
        realAUCs['s_mean'] = np.nan
        realAUCs['s_std'] = np.nan
        realAUCs['tuning'] = np.nan
        for (a,n), _ in realAUCs.groupby(['action','neuron']):
            dist = shuffledAUCs.loc[(a,n)].values
            auc_value = realAUCs.loc[(a,n), 'auc']
            realAUCs.loc[(a,n), 'pct'] = _getPctl(dist, auc_value)
            realAUCs.loc[(a,n), 's_mean'] = np.mean(dist)
            realAUCs.loc[(a,n), 's_std'] = np.std(dist)
            realAUCs.loc[(a,n), 'tuning'] = (auc_value - np.mean(dist)) / np.std(dist)
        realAUCs.reset_index(inplace=True)
        
        for k,v in [('date',sess.meta.date), ('animal',sess.meta.animal), 
                    ('genotype',sess.meta.genotype)]:
            realAUCs.insert(0,k,v)
        
        auc_df = auc_df.append(realAUCs, ignore_index=True)
    return auc_df


#%%
def getStSwRasterData(dataFile, popdf, action, sort_ascending=False):
    def _getPrevNextPhases(phase):
        trial = ['pS2C','mS2C','pC2S','mC2S','dS2C']
        side = 'L' if 'L' in phase else 'R'
        phaseNo = np.argmax(np.array(trial) == phase[:-2].replace(side, 'S'))
        if phase.endswith('.'):
            trial = [p.replace('S', side)+phase[-2:] for p in trial]
        if phase.endswith('!'):
            if phaseNo == 1:
                sides = ('L','R') if side == 'L' else ('R','L')
            if phaseNo in [2,3]:
                sides = ('R','L') if side == 'L' else ('L','R')
            trial = [p.replace('S', sides[0])+phase[-2:] for p in trial[:2]] + \
                    [p.replace('S', sides[1])+phase[-2:] for p in trial[2:]]
        return trial[phaseNo-1:phaseNo+2]
    
    piles = {action+'r.': [], action+'o.': [], action+'o!': []}    
    for (genotype, animal, date), auc in popdf.groupby(['genotype','animal','date']):
        sess = next(readSessions.findSessions(dataFile, genotype=genotype,
                                              animal=animal, date=date,
                                              task='2choice'))
        lfa = sess.labelFrameActions(reward="fullTrial", switch=True, splitCenter=True)
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)[list(auc.neuron)]
        
        for p in piles.keys():
            incl_phases = _getPrevNextPhases(p)
            X = deconv.loc[lfa.label.isin(incl_phases)]
            Y = lfa.loc[lfa.label.isin(incl_phases), ['label','actionProgress']]
            avgActivity = X.groupby([Y.label,(Y.actionProgress*5).astype("int")/5.0]).mean().T
            avgActivity = avgActivity[incl_phases]
            piles[p].append(avgActivity)
    
    for p in piles.keys():
        piles[p] = pd.concat(piles[p])
        
    stacked = pd.concat(list(piles.values()), keys=piles.keys(), axis=1).reset_index(drop=True)
    sort_idx = []
    for p in piles.keys():
        #sort_idx.append(stacked[(p,p)].max(axis=1) > .5)
        sort_idx.append(stacked[(p,p)].max(axis=1))
    sort_idx = pd.concat(sort_idx, axis=1)
    sort_idx = sort_idx.mean(axis=1)
    #stacked = stacked.loc[sort_idx.sort_values([0,1,2], ascending=False).index]
    stacked = stacked.loc[sort_idx.sort_values(ascending=sort_ascending).index]
    
    return stacked


#%% TODO: omg this is some horrible code :D
#def drawCoefficientWeightedAverage(dataFile, C, genotype, action, axes, cax=False,
#                                   shuffled=False):
#    C = (C.query('shuffled == @shuffled and genotype == @genotype and action == @action')
#          .set_index(['genotype','animal','date','action','neuron'])
#          .coefficient
#          .sort_index()
#          .copy())
#
#    # can't create a intensity plot without session data -> actually, looks like you can
#    s = next(readSessions.findSessions(dataFile, task='2choice'))
#    fvSt = fancyViz.SchematicIntensityPlot(s, splitReturns=False, splitCenter=True,
#                                           saturation=.5, linewidth=mpl.rcParams['axes.linewidth'])
#    fvSw = fancyViz.SchematicIntensityPlot(s, splitReturns=False, splitCenter=True,
#                                           saturation=.5, linewidth=mpl.rcParams['axes.linewidth'])
#    
#    for s in readSessions.findSessions(dataFile, genotype=genotype, task='2choice'):
#        deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
#        lfa = s.labelFrameActions(switch=True, reward='fullTrial')
#        if len(deconv) != len(lfa): continue
#        
#        coefs = C.loc[(s.meta.genotype,s.meta.animal,s.meta.date,action)]
#    
#        trans = (deconv * coefs).sum(axis=1) # svm normalizes each session in some way
##        trans -= trans.mean()
#        trans /= trans.std()
#        #print(trans.max())
#    
#        fvSt.setSession(s)
#        fvSt.setMask(lfa.label.str.endswith('r.'))
#        fvSt.addTraceToBuffer(trans)
#        fvSw.setSession(s)
#        fvSw.setMask(lfa.label.str.endswith('o!'))
#        fvSw.addTraceToBuffer(trans)
#    
#    stax, swax = axes[0], axes[1]
#    
#    fvSt.drawBuffer(ax=stax, cmap='RdYlGn') # drawing flushes buffer
#    img = fvSw.drawBuffer(ax=swax, cmap='RdYlGn')
#    
#    if cax:
#        cb = plt.colorbar(img, cax=cax)
#        cax.tick_params(axis='y', which='both',length=0)
#        cb.outline.set_visible(False)
#        

#%%
def drawPopAverageFV(dataFile, popdf, axes, cax=False, auc_weigh=False,
                     saturation=.25, smoothing=5, cmap='RdYlBu_r'):
    # can't create a intensity plot without session data -> not true!
    s = next(readSessions.findSessions(dataFile, task='2choice'))
    fvWSt = fancyViz.SchematicIntensityPlot(s, splitReturns=False, splitCenter=True,
                                            saturation=saturation, smoothing=smoothing,
                                            linewidth=mpl.rcParams['axes.linewidth'])
    fvLSt = fancyViz.SchematicIntensityPlot(s, splitReturns=False, splitCenter=True,
                                            saturation=saturation, smoothing=smoothing,
                                            linewidth=mpl.rcParams['axes.linewidth'])
    fvLSw = fancyViz.SchematicIntensityPlot(s, splitReturns=False, splitCenter=True,
                                            saturation=saturation, smoothing=smoothing,
                                            linewidth=mpl.rcParams['axes.linewidth'])
    
    # if weighing by auc, they need to sum to the total number of neurons for
    # the fancyViz average to make sense; assumes one auc value per neuron in popdf!
    if auc_weigh:
        popdf = popdf.copy()
        # .abs() in case of averaging stay & switch tuned neurons into a single plot
        # -> weights by absolute value, but maintains sign of tuning
        popdf['auc'] = (popdf.auc / popdf.auc.abs().sum()) * len(popdf.auc)
    
    for (genotype,animal,date), pop in popdf.groupby(['genotype','animal','date']):
        s = next(readSessions.findSessions(dataFile, task='2choice',
                                           genotype=genotype, animal=animal, date=date))
        
        deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        if auc_weigh:
            aucs = pop.set_index('neuron').auc
            deconv *= aucs
        
        lfa = s.labelFrameActions(switch=True, reward='fullTrial')
        # v obscure code for when the delay wasn't labeled properly
#        d_labels = ((lfa.set_index('actionNo').label.str.slice(0,5) + \
#                     lfa.groupby('actionNo').label.first().shift(1).str.slice(4))
#                    .reset_index().set_index(lfa.index))
#        lfa.loc[lfa.label.str.contains('d.$'), 'label'] = d_labels.fillna('-')
    
        fvWSt.setSession(s)
        fvWSt.setMask(lfa.label.str.endswith('r.'))
        for neuron in pop.neuron:
            fvWSt.addTraceToBuffer(deconv[neuron])

        fvLSt.setSession(s)
        fvLSt.setMask(lfa.label.str.endswith('o.'))
        for neuron in pop.neuron:
            fvLSt.addTraceToBuffer(deconv[neuron])
        
        fvLSw.setSession(s)
        fvLSw.setMask(lfa.label.str.endswith('o!'))
        for neuron in pop.neuron:
            fvLSw.addTraceToBuffer(deconv[neuron])
            
        s.hdfFile.close()
    
    wstax, lstax, lswax = axes[0], axes[1], axes[2]
    
    fvWSt.drawBuffer(ax=wstax, cmap=cmap) # drawing flushes buffer
    fvLSt.drawBuffer(ax=lstax, cmap=cmap)
    img = fvLSw.drawBuffer(ax=lswax, cmap=cmap)
    
    if cax:
        cb = plt.colorbar(img, cax=cax, orientation='horizontal')
        cax.tick_params(axis='x', which='both',length=0)
        cb.outline.set_visible(False)

