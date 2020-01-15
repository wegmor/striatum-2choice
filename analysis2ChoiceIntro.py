#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:40:29 2020

@author: mowe
"""

import numpy as np
import pandas as pd
from itertools import product
from utils import readSessions
from utils.cachedDataFrame import cachedDataFrame


#%% TODO: this could be done with the data from the function below (getChoiceData)
def getPeriSwapChoices(dataFile, win_size):
    swap_df = pd.DataFrame() # frame holds choice data around outcome swap
    
    for s in readSessions.findSessions(dataFile, task='2choice',
                                       onlyRecordedTrials=False):
        # load behavior data
        df = s.readSensorValues(slim=False, onlyRecording=False)
        
        # left entry -> beamL increments from 0 to 1, etc.
        df['leftIn'] = df.beamL.diff() == 1
        df['rightIn'] = df.beamR.diff() == 1
        
        # reduce df to choice port entries
        df = (df.loc[df.leftIn | df.rightIn,
                     ['animal','leftIn','switchNo','rewardP']]
                .reset_index(drop=True))
        
        # get peri-swap windows
        swap_idx = df[df.switchNo.diff() == 1].index # first choice after swap
        swap_windows = [(win[['animal','leftIn','rewardP','switchNo']]
                            .set_index(np.arange(-win_size[0], win_size[1])))
                            for win in [df.loc[idx-win_size[0]:idx+win_size[1]-1] for idx in swap_idx]
                            if len(win) == np.sum(win_size)]
        swap_windows = pd.concat(swap_windows, keys=np.arange(len(swap_windows)),
                                 names=['swap'])
        swap_windows = swap_windows.reset_index(-1).rename(columns={'level_1':'swapDist'})
        
        swap_windows['leftIn'] = swap_windows.leftIn.astype('float')
        # v set rewardP for swap window to initial port
        swap_windows['rewardP'] = swap_windows.loc[swap_windows.swapDist == -1, 'rewardP']
        
        # discard choices made in an earlier / later swap
        include = (swap_windows.groupby('swap')
                       .apply(lambda sw:
                                  (sw.switchNo.isin(sw.loc[sw.swapDist.isin([-1,0]),'switchNo'].values)
                                     .values)))
        swap_windows = swap_windows[np.concatenate(include)]
        swap_windows = swap_windows.drop(columns=['switchNo'])
        
        swap_df = swap_df.append(swap_windows.reset_index(drop=True), ignore_index=True)
    
    swap_df = swap_df.groupby(['animal','rewardP','swapDist'])[['leftIn']].mean()
    return swap_df


#%%
def getChoiceData(dataFile):
    choice_df = pd.DataFrame()
    
    for s in readSessions.findSessions(dataFile, task='2choice',
                                       onlyRecordedTrials=False):
        # load behavior data
        df = s.readSensorValues(slim=False, onlyRecording=False)
        
        # side exits
        df['leftEx'] = df.beamL.diff() == -1
        df['rightEx'] = df.beamR.diff() == -1
        
        # side triggered
        df['triggered'] = df.ledL.diff() == -1
        df['triggerNo'] = df.triggered.cumsum()
        
        # reduce df to choice port exits
        df = df.loc[df.leftEx | df.rightEx,
                    ['animal','date','leftEx','rightEx',
                     'rewardNo','triggerNo','rewardP']].copy()
        
        # reward port switches prior to port exit
        df['rewardP'] = df.rewardP.shift(1)
        
        # define reward -- it is delivered when the beam is still broken,
        # after 350 ms delay, before port exit
        df['reward'] = (df.rewardNo.diff() >= 1).astype('bool')
        df['rewardx'] = df.rewardNo.diff()
        df.loc[~df.reward, 'rewardx'] = np.nan
        df['rewardx'] = df.rewardx.fillna(method='ffill')
        df['rewardNo'] = df.reward.cumsum() # recompute in case trials have been dropped
        df['correct'] = ((df.leftEx & (df.rewardP == 'L')) |
                         (df.rightEx & (df.rewardP == 'R')))
        
        # label trials in which the side port was activated
        # need to redo this because light will switch off prior to port exit
        df['triggered'] = df.triggerNo.diff() >= 1
    
        # convert to int
        df['leftEx'] = df.leftEx.astype('int')
        df['rightEx'] = df.rightEx.astype('int')
        df['correct'] = df.correct.astype('int')
        df['reward'] = df.reward.astype('int')
        df['triggered'] = df.triggered.astype('int')
        
        # label switch trials
        df['switch'] = (df.leftEx.diff().abs() == 1).astype('int')
        # labels 1st entry into a port, unlike '!', which labels last side entry!!
        df['switchNo'] = df.switch.cumsum()
        
        df = df.dropna() # diffs/shift(1) lead to NaN in first row
        
        # count trials without a reward
        df['sinceReward'] = df.groupby('rewardNo').cumcount()
        df['toReward'] = -(df.groupby('rewardNo').cumcount(ascending=False).shift(1))
        df.loc[df.rewardNo == 0, 'sinceReward'] = np.nan
        df.loc[df.rewardNo == df.rewardNo.iloc[-1], 'toReward'] = np.nan
        df.loc[df.sinceReward == 0, 'toReward'] = 0  # otherwise last switch is lost in "toReward"
        
        # 'entry distance' from next switch
        df['sinceSwitch'] = df.groupby('switchNo').cumcount()
        df['toSwitch'] = -(df.groupby('switchNo').cumcount(ascending=False).shift(1))
        df.loc[df.switchNo == 0, 'sinceSwitch'] = np.nan
        df.loc[df.switchNo == df.switchNo.iloc[-1], 'toSwitch'] = np.nan
        df.loc[df.sinceSwitch == 0, 'toSwitch'] = 0
        
        choice_df = choice_df.append(df, ignore_index=True)
    
    return choice_df


#%% https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_xaxis(ax1, v1, ax2, v2):
    """adjust ax2 xlimit so that v2 in ax2 is aligned to v1 in ax1"""
    x1, _ = ax1.transData.transform((v1, 0))
    x2, _ = ax2.transData.transform((v2, 0))
    inv = ax2.transData.inverted()
    dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0))
    minx, maxx = ax2.get_xlim()
    ax2.set_xlim(minx+dx, maxx+dx)
    

#%% for raster plot
def getActionWindows(sess, win_size=(8, 7), nan_actions=False,
                     incl_actions=["pL2C","pR2C","mL2C","mR2C","pC2L","pC2R",
                                   "mC2L","mC2R","dL2C","dR2C"],
                     incl_trialTypes=["r.","o.","o!"]):
    lfa = sess.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
    deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    
    if len(lfa) != len(deconv):
        return(pd.DataFrame)
    
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

@cachedDataFrame("introRasterAverages.pkl")
def getRasterAverages(dataFile, smooth=True):
    rasterAvgs = pd.DataFrame()
    for sess in readSessions.findSessions(dataFile, task='2choice'):
        windows = getActionWindows(sess)
        if windows.empty: continue
        windows.set_index(['neuron','action','trialType','actionNo'],
                          inplace=True)
        windows = windows['frameNo']
        if smooth:
            windows = windows.rolling(3, center=True, axis=1).mean()
        
        avgs = windows.groupby(['neuron','action','trialType']).mean()
        avgs = avgs.dropna(axis=1)
        
        for k,v in [('date',sess.meta.date),('animal',sess.meta.animal),
                    ('genotype',sess.meta.genotype)]:
            avgs.insert(0,k,v)
            
        rasterAvgs = rasterAvgs.append(avgs, ignore_index=True)
    return rasterAvgs


#%% for trajectory plot
@cachedDataFrame("introTrajectoryDF.pkl")
def getTrajectoryData(dataFile):
    Xs = []
    for sess in readSessions.findSessions(dataFile, task='2choice'):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        labels = sess.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
        if len(deconv) != len(labels): continue
    
        # trials start upon outcome delivery
        labels['trialNo'] = (labels.label.str.contains('p[LR]2C[or][\.\!]$') * \
                             (labels.actionNo.diff() == 1).astype('int')).cumsum()
        labels = labels.set_index('trialNo')
        labels['trialType'] = labels.groupby('trialNo').label.first()
        labels['bin'] = labels.actionProgress // (1/4)
        labels = labels.reset_index().set_index(['trialType','trialNo'])
        labels = labels.sort_index()
        # only keep actions with all actions longer than 5 frames
        labels['include'] = (labels.groupby(['trialType','trialNo']).actionDuration.apply(
                                 lambda t: (np.unique(t) >= 5).all()))
        # make sure trials included follow trial structure perfectly
        properTrials = [l1+l2 for l1,l2 in product((['pL2C','mL2C'],['pR2C','mR2C']),
                                                   (['pC2L','mC2L','dL2C'],['pC2R','mC2R','dR2C']))]
        labels['include'] = (labels.include & (labels.groupby(['trialType','trialNo','actionNo']).label.first()
                                                     .groupby(['trialType','trialNo'])
                                                     .apply(lambda t: [l[:-2] for l in t] in properTrials)))
    
        labels = labels.sort_values(['actionNo','actionProgress']) # same sorting as in deconv
        labels = labels.reset_index().set_index(['include','trialType','trialNo','actionNo','bin'])
    
        # mean activity for bins
        deconv.columns.name = 'neuron'
        deconv = deconv.set_index(labels.index).sort_index()
        X = deconv.loc[True].groupby(['trialType','trialNo','actionNo','bin']).mean()
        X['bin'] = X.groupby(['trialType','trialNo']).cumcount() # full trial bins
        X = X.reset_index(['actionNo','bin'], drop=True).set_index('bin', append=True)
        X = X.loc[['pL2Cr.','pL2Co.','pL2Co!',
                   'pR2Cr.','pR2Co.','pR2Co!']]
        X = X.unstack('bin')
        X = X.dropna() # drop trials with nans
        
        # if there are fewer than 20 trials in any trial type, omit! -> smallest # trials determines
        # the size of the final collated data set
        if not (X.groupby('trialType').size() >= 20).all():
            continue
    
        X['trialNo'] = X.groupby('trialType').cumcount() # continuous trial nos / label
        X = X.reset_index('trialNo', drop=True).set_index('trialNo', append=True)
        X = X.stack('neuron')
        
        for k,v in [('genotype',sess.meta.genotype), ('animal',sess.meta.animal),
                    ('date',sess.meta.date)]:
            X.insert(0,k,v)
        X = X.set_index(['genotype','animal','date'], append=True)
        X = X.reorder_levels([3,4,5,2,0,1])
        
        Xs.append(X)
    
    X = pd.concat(Xs, axis=0)
    return X


def prepTrajectoryData(X, trials=None, shuffle=True, seed=None):
    T = X.copy()
    # shuffle trial numbers / neuron
    if seed:
        np.random.seed(seed)
    if shuffle:
        T['trialNo'] = (T.groupby(['trialType','genotype','animal','date','neuron'], as_index=False)
                         .apply(lambda g: pd.Series(np.random.permutation(len(g)), index=g.index))
                         .reset_index(0, drop=True))
        T = T.reset_index('trialNo', drop=True).set_index('trialNo', append=True)
    # reformat, reduce to "complete" trials
    T = T.unstack(['genotype','animal','date','neuron']).stack('bin')
    T = T.loc[:,(T != np.inf).all()].dropna().copy()
    # reduce to x trials per condition
    if trials:
        T = T.query('trialNo < @trials').copy()
    # normalize matrix
    T -= T.mean(axis=1).values[:,np.newaxis]
    T /= T.std(axis=1).values[:,np.newaxis]
    return T


