#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:45:29 2019

@author: mowe
"""

import numpy as np
import pandas as pd
from utils import readSessions


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