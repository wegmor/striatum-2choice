#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 02:23:24 2020

@author: mowe
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.cachedDataFrame import cachedDataFrame
from utils import readSessions
import style
import seaborn as sns
import itertools


def plot2CTrackingEvent(track, ax, color='k', lw=.5, alpha=1., portsUp=True):
    body = track['body'][['x','y']]
    head = (track['leftEar'][['x','y']] + track['rightEar'][['x','y']]) / 2
    tail = track['tailBase'][['x','y']]
    
    t = ax.transData
    if portsUp:
        t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(15/2, 15/2, 90) + t
    
    for n in range(len(head)):
        ax.plot(np.stack([head.iloc[n].x, body.iloc[n].x, tail.iloc[n].x]),
                np.stack([head.iloc[n].y, body.iloc[n].y, tail.iloc[n].y]),
                color=color, lw=lw, alpha=alpha,
                #alpha=np.clip(((len(head)-n)/len(head))**.8,0,1)*alpha,
                clip_on=False, transform=t)


def get2CTrajectoryDists(trajectories, cache_prefix):
    def _getDist(track1, track2):
        track1, track2 = track1.copy(), track2.copy()
        head1 = (track1['leftEar'][['x','y']] + track1['rightEar'][['x','y']]) / 2
        head2 = (track2['leftEar'][['x','y']] + track2['rightEar'][['x','y']]) / 2
        track1[('head','x')], track1[('head','y')] = head1.x, head1.y
        track2[('head','x')], track2[('head','y')] = head2.x, head2.y
        track1 = (track1.loc[:,track1.columns.get_level_values(1).isin(['x','y'])]
                        .loc[:,['body','head','tailBase']]
                        .stack(0))
        track2 = (track2.loc[:,track2.columns.get_level_values(1).isin(['x','y'])]
                        .loc[:,['body','head','tailBase']]
                        .stack(0))
        dists = ((track1-track2)**2).sum(axis=1, skipna=False).map(np.sqrt).unstack(-1).mean().mean()
        return dists
    
    @cachedDataFrame('{}_trajectories.pkl'.format(cache_prefix))
    def _aux(trajectories):
        trajectories = trajectories.copy()
        action2behavior = trajectories.groupby('actionNo').behavior.first().str.slice(-2)
        trajectories = (trajectories.groupby(['actionNo','bin']).mean()
                                    [['body','tailBase','leftEar','rightEar']])
        
        pairIdxs = itertools.combinations(trajectories.index.get_level_values(0).unique(), 2)
        out = []
        for idx in pairIdxs:
            dist = _getDist(trajectories.loc[idx[0]], trajectories.loc[idx[1]])
            tts = '{}X{}'.format(*sorted([action2behavior.loc[idx[0]], action2behavior.loc[idx[1]]]))
            out.append([tts, dist])
        out = pd.DataFrame(out, columns=['trialTypes','distance']) 
        return out
    
    return _aux(trajectories)
    
    
def getActionMeans(endoDataPath, genotype, animal, date):
    @cachedDataFrame('{}_{}_{}_actionMeans.pkl'.format(genotype, animal, date))
    def _aux():
        s = next(readSessions.findSessions(endoDataPath, task='2choice',
                                           genotype=genotype, animal=animal, date=date))
        lfa = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
        deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    
        if not len(lfa) == len(deconv):
            print(str(s)+': more labeled frames than signal!')
            return -1
    
        means = pd.DataFrame(deconv.groupby([lfa['label'], lfa['actionNo']]).mean().stack(),
                             columns=['trialMean'])
        means.index.names = means.index.names[:2] + ['neuron']
        means.reset_index(inplace=True)
        means['action'] = means.label.str.slice(0,4)
    
        return(means)
    return _aux()



def getActionWindows(endoDataPath, genotype, animal, date,
                     win_size=(20, 19), nan_actions=True,
                     incl_actions=["pL2C","pR2C","mL2C","mR2C","pC2L","pC2R",
                                   "mC2L","mC2R","dL2C","dR2C"],
                     incl_trialTypes=["r.","o.","o!"]):
    @cachedDataFrame('{}_{}_{}_actionWindows.pkl'.format(genotype, animal, date))
    def _aux():
        s = next(readSessions.findSessions(endoDataPath, task='2choice',
                                           genotype=genotype, animal=animal, date=date))
        lfa = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
        deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        
        if not len(lfa) == len(deconv):
            print(str(s)+': more labeled frames than signal!')
            return -1
    
        lfa['index'] = deconv.index
        deconv = deconv.set_index([lfa.label,lfa.actionNo], append=True)
        deconv.index.names = ['index','label','actionNo']
        deconv.columns.name = 'neuron'
        
        lfa = lfa.loc[lfa.label.str.slice(0,4).isin(incl_actions) &
                      lfa.label.str.slice(4).isin(incl_trialTypes)]
        actions_idx = lfa.groupby('actionNo')[['index','actionNo','label']].first().values
        
        windows = []
        neurons = deconv.columns
        for idx, actionNo, label in actions_idx:
            win = deconv.loc[idx-win_size[0]:idx+win_size[1]].reset_index()
            if nan_actions:
                win.loc[win.actionNo > actionNo, neurons] = np.nan
                win.loc[win.actionNo < actionNo-1, neurons] = np.nan
            win['frameNo'] = np.arange(len(win))
            win['label'] = label
            win['actionNo'] = actionNo
            win = win.set_index(['actionNo','label','frameNo'])[neurons]
            win = win.unstack('frameNo').stack('neuron')
            win.columns = pd.MultiIndex.from_product([['frameNo'], win.columns])
            windows.append(win.reset_index())
        windows = pd.concat(windows, ignore_index=True)
        windows['action'] = windows.label.str.slice(0,4)
        windows['trialType'] = windows.label.str.slice(4)
        
        return windows
    return _aux()


def getPhaseLabels(phase):
    if 'S' in phase:
        actions = [phase.replace('S', s) for s in 'LR']
        inclLabels = [actions[0]+tt for tt in ['r.','o.','o!']] + \
                     [actions[1]+tt for tt in ['o!','o.','r.']]
    else:
        inclLabels = [phase+tt for tt in ['r.','o.','o!']]
    return inclLabels
        
