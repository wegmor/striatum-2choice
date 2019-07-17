#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:24:45 2019

@author: mowe
"""

import numpy as np
import pandas as pd
from utils import readSessions


#%%
def get_centers(rois):
    # find pixel of maximum intensity in each mask; use as neuron center
    centers = np.array(np.unravel_index(np.array([np.argmax(roi) for roi in rois]),
                                                  rois.shape[1:]))
    centers = centers[::-1].T
    return(centers)
    
def getActionAverages(traces, apf):
    keepLabels = ['pC2L-', 'mC2L-',
                  'pC2R-', 'mC2R-',
                  'pL2Cd', 'pL2Co', 'pL2Cr', 'mL2C-',
                  'pR2Cd', 'pR2Co', 'pR2Cr', 'mR2C-']
    apf = apf.loc[apf.label.isin(keepLabels)].copy()
    apf['label'] = apf.label.astype('str')
    actionAvg = traces.loc[apf.index].groupby([apf.label,apf.actionNo]).mean().dropna()
    labels = apf.groupby(['label','actionNo']).first().loc[actionAvg.index,
                                                           ['actionDuration']]    
    return(actionAvg, labels)

def wAvg(group, var, weights):
    return(np.average(group[var], weights=group[weights]))
    
def bootstrap(group, var, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(np.arange(len(group[var])), size=len(group[var]),
                               replace=True)
        avgs.append(np.average(group[var].iloc[idx], weights=group[weights].iloc[idx]))
    return(np.std(avgs))
    
def jitter(x, std):
    return(x+np.random.normal(0,std,size=len(x)))
    
    
#%%
def getTuningData(dataFilePath, no_shuffles=1000):
    df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='2choice'):
        traces = s.readDeconvolvedTraces(zScore=True).reset_index(drop=True) # frame no as index
        apf = s.labelFrameActions(switch=False, reward='sidePorts',
                                  splitCenter=True).reset_index(drop=True)
        
        # TODO: fix remaining recordings with dropped frames
        if traces.shape[0] != apf.shape[0]:
            continue
        
        actionAvg, labels = getActionAverages(traces, apf)  # mean per action
        actionAvg = actionAvg.groupby('label').mean()       # mean per label

        shuffle_labels = s.shuffleFrameLabels(no_shuffles, switch=False,
                                              reward='sidePorts', splitCenter=True)
        
        s_actionAvgs = []
        for s_apf in shuffle_labels:
            s_actionAvg, s_labels = getActionAverages(traces, s_apf)
            s_actionAvg = s_actionAvg.groupby('label').mean()
            s_actionAvgs.append(s_actionAvg)
        s_actionAvgs = pd.concat(s_actionAvgs, keys=np.arange(no_shuffles),
                                 names=['shuffle']).reorder_levels(['label','shuffle'])

        for action, adata in s_actionAvgs.groupby('label'):
            for neuron in adata:
                ndict = {}
                dist = adata[neuron].values # shuffled "label" means distribution
                value = actionAvg.loc[action, neuron] # actual mean
                
                ndict['genotype'] = s.meta.genotype
                ndict['animal'] = s.meta.animal
                ndict['date'] = s.meta.date
                ndict['neuron'] = neuron
                ndict['action'] = action
                ndict['mean'] = value
                ndict['s_mean'] = dist.mean()
                ndict['s_std'] = dist.std()
                ndict['tuning'] = (ndict['mean'] - ndict['s_mean']) / ndict['s_std']
                # v percentile of the actual mean in the shuffled distribution
                ndict['pct'] = np.searchsorted(np.sort(dist), value) / len(dist)
                
                df = df.append(pd.Series(ndict), ignore_index=True)
        
    return df


#%%
