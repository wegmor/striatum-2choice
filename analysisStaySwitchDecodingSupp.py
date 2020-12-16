#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 02:23:24 2020

@author: mowe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.cachedDataFrame import cachedDataFrame
from utils import readSessions
import tqdm
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


def _getStSwCodingDirectionAvgs(endoDataPath, binPhases=False, resampleAUCs=False):
    def _resampleSessAUCs(aucs):
        aucs = aucs.set_index(aucs.neuron).copy()
        sessNeurons = aucs.neuron.unique()
        resampled = aucs.loc[np.random.choice(sessNeurons, size=len(sessNeurons), replace=True)]
        resampled.reset_index(drop=True)
        return resampled
    
    # compute coding direction weights (stay-switch AUC-based)
    incl_actions=['pL2C','mL2C','pC2L','mC2L','dL2C',#'dR2C','mC2R','pC2R','mR2C','pR2C']
                  'pR2C','mR2C','pC2R','mC2R','dR2C']
    AUCS = pd.read_pickle('cache/staySwitchAUC.pkl') # TODO: shouldn't load that directly
    if resampleAUCs: # resample for bootstrapping
        AUCS = AUCS.groupby(['genotype','animal','date'], group_keys=False).apply(_resampleSessAUCs)
    AUCS = AUCS.set_index(['genotype','action']).sort_index()
    #AUCS.loc[~((AUCS.pct < .005) | (AUCS.pct > .995)), 'auc'] = 0  # zero non-significant aucs
    AUCS['auc'] /= AUCS['auc'].abs().groupby(['genotype','action']).sum() # absolute aucs pooled over all neurons sum to 1
    AUCS = AUCS.reset_index().set_index(['genotype','animal','date','action','neuron']).sort_index()
    AUCS = AUCS[['auc']]
    
    # get auc-weighted means, i.e. "coding direction"
    wAM = pd.DataFrame()
    for sess in tqdm.tqdm(readSessions.findSessions(endoDataPath, task='2choice')):
        try:
            aucs = AUCS.loc[sess.meta.genotype,sess.meta.animal,sess.meta.date].copy()
        except KeyError:
            print('no AUCs available, skip!')
            continue
        
        # compute, per neuron, (action | trial type) means and subtract overall action mean -> trial type-specific activity
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        deconv.columns.name = 'neuron'
        lfa = sess.labelFrameActions(reward='fullTrial', switch=True).reset_index(drop=True)
        if binPhases:
            lfa['bin'] = lfa.actionProgress // (1/5)
            actionMeans = deconv.groupby([lfa['label'],lfa['actionNo'],lfa['bin']]).mean()
        else:
            actionMeans = deconv.groupby([lfa.label,lfa.actionNo]).mean()
        acIndex = pd.Index(actionMeans.index.get_level_values(0).str.slice(0,-2), name='action')
        ttIndex = pd.Index(actionMeans.index.get_level_values(0).str.slice(-2), name='trialType')
        actionMeans = actionMeans.set_index([acIndex,ttIndex], append=True).reset_index('label',drop=True)
        if binPhases:
            actionMeans.index = actionMeans.index.reorder_levels((1,2,3,0))
        else: 
            actionMeans.index = actionMeans.index.reorder_levels((1,2,0))
        actionMeans = actionMeans.query('action in @incl_actions & trialType in ["r.","o.","o!"]')
        if binPhases:
            meanSubActionMeans = (actionMeans.groupby(['action','bin','trialType']).mean() - \
                                  actionMeans.groupby(['action','bin']).mean())
        else:
            meanSubActionMeans = (actionMeans.groupby(['action','trialType']).mean() - \
                                  actionMeans.groupby('action').mean())
    
        # AUC-weigh means
        wMeans = []
        for action in incl_actions:
            # v make sure to use resampled set of neurons
            wMeanSubActionMeans = meanSubActionMeans.loc[:,aucs.loc[action].index] * aucs.loc[action].values.T
            wMeanSubActionMeans = pd.DataFrame(wMeanSubActionMeans.stack(0), columns=['meanF'])
            wMeanSubActionMeans['tuning'] = action
            wMeans.append(wMeanSubActionMeans.reset_index())
        wMeans = pd.concat(wMeans)
    
        for k,v in [('genotype',sess.meta.genotype), ('animal',sess.meta.animal), ('date',sess.meta.date)]:
            wMeans.insert(0,k,v)
        
        wAM = wAM.append(wMeans, ignore_index=True)
    
    wAM['action'] = pd.Categorical(wAM['action'], incl_actions, ordered=True)
    wAM['tuning'] = pd.Categorical(wAM['tuning'], incl_actions, ordered=True)
    wAM['trialType'] = pd.Categorical(wAM['trialType'], ['r.','o.','o!'], ordered=True)
    
    # needs to be sum since weights add up to 1 and not number of neurons
    if binPhases:
        cdMeans = wAM.groupby(['genotype','tuning','action','trialType','bin'])[['meanF']].sum().unstack('tuning')['meanF']
        cdMeans.index = cdMeans.index.reorder_levels((0,2,1,3))
    else:
        cdMeans = wAM.groupby(['genotype','tuning','action','trialType'])[['meanF']].sum().unstack('tuning')['meanF']
        cdMeans.index = cdMeans.index.reorder_levels((0,2,1))
    cdMeans.sort_index(axis=0, inplace=True)
    
    return cdMeans


@cachedDataFrame('stSwCodingDirectionRaster.pkl')
def getStSwCodingDirectionRaster(endoDataPath):
    return _getStSwCodingDirectionAvgs(endoDataPath, binPhases=False, resampleAUCs=False)


@cachedDataFrame('stSwCodingDirectionTraces.pkl')
def getStSwCodingDirectionTraces(endoDataPath, n_bootstrap=1000):
    actualTraces = _getStSwCodingDirectionAvgs(endoDataPath, binPhases=True, resampleAUCs=False)
    resampledTraces = [_getStSwCodingDirectionAvgs(endoDataPath, binPhases=True, resampleAUCs=True) for n in range(n_bootstrap)]
    return actualTraces, resampledTraces
    
    