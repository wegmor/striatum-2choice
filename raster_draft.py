#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:54:33 2020

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import figurefirst
import style
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import analysisStaySwitchDecoding
import analysisTunings
from scipy.spatial.distance import squareform, pdist
from collections import defaultdict
from utils import readSessions, fancyViz
import cmocean
from matplotlib.backends.backend_pdf import PdfPages
plt.ioff()
style.set_context()

#%%
endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path("data") / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#%%
def getActionWindows(win_size=(20, 19), nan_actions=True,
                     incl_actions=["pL2C","pR2C","mL2C","mR2C","pC2L","pC2R",
                                   "mC2L","mC2R","dL2C","dR2C"],
                     incl_trialTypes=["r.","o.","o!"]):
    windows = pd.DataFrame()
    for s in readSessions.findSessions(endoDataPath, task='2choice'):
        print(str(s))
        lfa = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
        deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        
        if not len(lfa) == len(deconv):
            print(str(s)+': more labeled frames than signal!')
            continue
    
        lfa['index'] = deconv.index
        deconv = deconv.set_index([lfa.label,lfa.actionNo], append=True)
        deconv.index.names = ['index','label','actionNo']
        deconv.columns.name = 'neuron'
        
        lfa = lfa.loc[lfa.label.str.slice(0,4).isin(incl_actions) &
                      lfa.label.str.slice(4).isin(incl_trialTypes)]
        actions_idx = lfa.groupby('actionNo')[['index','actionNo','label']].first().values
        
        _windows = []
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
            _windows.append(win.reset_index())
        _windows = pd.concat(_windows, ignore_index=True)
        
        for k,v in [('date',s.meta.date),('animal',s.meta.animal),('genotype',s.meta.genotype)]:
            _windows.insert(0,k,v)
            
        windows = windows.append(_windows, ignore_index=True)
    
    windows['action'] = windows.label.str.slice(0,4)
    windows['trialType'] = windows.label.str.slice(4)
    return windows

#%%
windows = getActionWindows(win_size=(8,7), nan_actions=False)
windows.set_index(['genotype','animal','date','neuron','action','trialType','actionNo'],
                  inplace=True)
windows = windows['frameNo']
windows = windows.rolling(3, center=True, axis=1).mean()

mean_rs = windows.groupby(['genotype','animal','date','neuron','action','trialType']).mean()
mean_rs = mean_rs.dropna(axis=1)
mean_rs.to_pickle('actionAvgs_raster.pkl')

del windows

#%%
mean_rs = pd.read_pickle('actionAvgs_raster.pkl')

#%%
trialType = 'o.'
df = mean_rs.query('trialType == @trialType').copy().reset_index('trialType', drop=True)
df = df.reset_index('action')
if trialType.endswith('.'):
    df['action'] = pd.Categorical(df.action, categories=['pL2C','mL2C','pC2L','mC2L','dL2C',
                                                         'pR2C','mR2C','pC2R','mC2R','dR2C'],
                                  ordered=True)
else:
    df['action'] = pd.Categorical(df.action, categories=['pL2C','mL2C','pC2R','mC2R','dR2C',
                                                         'pR2C','mR2C','pC2L','mC2L','dL2C'],
                                  ordered=True)
df = df.set_index('action', append=True).unstack('action')
df.columns = df.columns.reorder_levels([1,0])
df = df.sort_index(axis=1).sort_index(axis=0)

df /= mean_rs.unstack(['action','trialType']).abs().max(axis=1).values[:,np.newaxis]
#df /= df.abs().max(axis=1).values[:,np.newaxis]

#%%
idx = pd.DataFrame({'idxmin':np.argmin(df.values, axis=1), 
                    'idxmax':np.argmax(df.values, axis=1), 
                    'posPeak':(df.max(axis=1) > df.min(axis=1).abs()).values},
                   index=df.index)

def getSorting(row, no_cols):
    return (row['idxmax'] - no_cols) if row.posPeak else (np.abs(row['idxmin'] - no_cols))

idx['sorting'] = idx.apply(getSorting, axis=1, no_cols=df.shape[1])
idx['right'] = idx.sorting.abs() <= (df.shape[1] // 2)
idx = idx.sort_values(['right','sorting'])

#%%
posCount = pd.DataFrame(idx.loc[idx.posPeak].groupby(['genotype','idxmax']).size(),
                        columns=['count']).reset_index()
posCount['bin'] = posCount['idxmax'] // (df.shape[1] // 10 // 2)
posCount = posCount.groupby(['genotype','bin'])[['count']].sum()
posCount['count'] /= idx.groupby('genotype').size()
posCount = posCount.unstack('genotype')['count']
posCount = posCount.reindex(np.arange(df.shape[1] // (df.shape[1] // 10 // 2)))
posCount['action'] = np.stack([df.columns.levels[0], df.columns.levels[0]], axis=1).flatten()
posCount.reset_index(inplace=True)
posCount['bin'] = posCount['bin'] % 2
posCount.set_index(['action','bin'], inplace=True)
posCount = posCount[['d1','a2a','oprm1']]

#%%
df = df.loc[idx.index]

fig, axs = plt.subplots(2, 10, figsize=(5,5),
                        gridspec_kw={'height_ratios':(.15, .85),
                                     'hspace':0.02})

barcol = [style.getColor(gt) for gt in posCount.columns]
for p, action in enumerate(df.columns.levels[0]):
    axs[0,p].bar(np.arange(-3,0), posCount.loc[action,0].values, color=barcol)
    axs[0,p].bar(np.arange(1,4), posCount.loc[action,1].values, color=barcol)
    axs[0,p].axvline(0, c='k', ls=':')
    
    axs[1,p].imshow(df.loc[:,action].values, aspect='auto', vmin=-1, vmax=1, cmap='RdBu_r')
    axs[1,p].axvline(df.loc[:,action].shape[1] // 2 - 1 + .5, c='k', ls=':')
    
for ax in axs[0]:
    ax.set_ylim((0,.2))
    ax.axis('off')
axs[0,0].axis('on')
axs[0,0].set_xticks(())
axs[0,0].set_yticks((0,.1,.2))
axs[0,0].set_yticklabels((0,10,20))
axs[0,0].set_yticks((.05,.15), minor=True)
sns.despine(bottom=True, ax=axs[0,0])
    
for ax in axs[1]:
    ax.axis('off')

fig.savefig('svg/raster.svg', bbox_inches='tight', pad_inches=0)

