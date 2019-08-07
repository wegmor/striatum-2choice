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
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import readSessions, fancyViz
import statsmodels.api as sm
import style

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

                    for baseLabel in ("pC2L", "mC2L", "pC2R", "mC2R", "mL2C", "mR2C"):
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
        deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
        lfa = sess.labelFrameActions(reward='fullTrial', switch=True)
        if len(deconv) != len(lfa): continue
        slfa = sess.shuffleFrameLabels(reward='fullTrial', switch=True)[0]
        selectedLabels = [base+trial for base in ['mL2C','mC2L','pC2L',
                                                  'pC2R','mC2R','mR2C']
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


def getActionValues(dataFile):
    coefficients, regression_df = getAVCoefficients(dataFile)
    
    actionValues = pd.DataFrame()   
    for sess in readSessions.findSessions(dataFile, task='2choice'):
        lfa = sess.labelFrameActions(reward='fullTrial', switch=True)
        sv = sess.readSensorValues()
        dummy_df, reg_vars = getRegressionVars(sv)
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
    return actionValues, coefficients, regression_df


#%% TODO: omg this is some horrible code :D
def drawCoefficientWeightedAverage(dataFile, C, genotype, action, axes, cax=False,
                                   shuffled=False):
    C = (C.query('shuffled == @shuffled and genotype == @genotype and action == @action')
          .set_index(['genotype','animal','date','action','neuron'])
          .coefficient
          .sort_index()
          .copy())

    # can't create a intensity plot without session data
    s = next(readSessions.findSessions(dataFile, task='2choice'))
    fvSt = fancyViz.SchematicIntensityPlot(s, splitReturns=False, splitCenter=True,
                                           saturation=.5, linewidth=mpl.rcParams['axes.linewidth'])
    fvSw = fancyViz.SchematicIntensityPlot(s, splitReturns=False, splitCenter=True,
                                           saturation=.5, linewidth=mpl.rcParams['axes.linewidth'])
    
    for s in readSessions.findSessions(dataFile, genotype=genotype, task='2choice'):
        deconv = s.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
        lfa = s.labelFrameActions(switch=True, reward='fullTrial')
        if len(deconv) != len(lfa): continue
        
        coefs = C.loc[(s.meta.genotype,s.meta.animal,s.meta.date,action)]
    
        trans = (deconv * coefs).sum(axis=1) # svm normalizes each session in some way
#        trans -= trans.mean()
        trans /= trans.std()
        #print(trans.max())
    
        fvSt.setSession(s)
        fvSt.setMask(lfa.label.str.endswith('r.'))
        fvSt.addTraceToBuffer(trans)
        fvSw.setSession(s)
        fvSw.setMask(lfa.label.str.endswith('o!'))
        fvSw.addTraceToBuffer(trans)
    
    stax, swax = axes[0], axes[1]
    
    fvSt.drawBuffer(ax=stax, cmap='RdYlGn') # drawing flushes buffer
    img = fvSw.drawBuffer(ax=swax, cmap='RdYlGn')
    
    if cax:
        cb = plt.colorbar(img, cax=cax)
        cax.tick_params(axis='y', which='both',length=0)
        cb.outline.set_visible(False) 