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
trialType = 'r.'
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
layout = figurefirst.FigureLayout(templateFolder / "2choiceIntro.svg")
layout.make_mplfigures()

#%%
df = df.loc[idx.index]

barcol = [style.getColor(gt) for gt in posCount.columns]
no_bins = df['pL2C'].shape[1]
for action in df.columns.levels[0]:
    barAx = layout.axes['bar_{}'.format(action)]['axis']
    barAx.bar(np.arange(-3,0), posCount.loc[action,0].values, color=barcol)
    barAx.bar(np.arange(1,4), posCount.loc[action,1].values, color=barcol)
    barAx.axvline(0, c='k', ls=':')
    barAx.set_ylim((0,.2))
    barAx.axis('off')
    if action == 'dR2C':
        barAx.axis('on')
        barAx.set_xticks(())
        sns.despine(bottom=True, left=True, right=False, ax=barAx,
                    offset=1)
        barAx.set_yticks((0,.1,.2))
        barAx.set_yticklabels((0,10,20))
        barAx.set_yticks((.05, .15), minor=True)
        barAx.set_ylabel('% neurons')
        barAx.yaxis.set_label_position('right')
        barAx.yaxis.set_ticks_position('right')
    
    rasterAx = layout.axes['raster_{}'.format(action)]['axis']
    img = rasterAx.imshow(df.loc[:,action].values, aspect='auto',
                          vmin=-1, vmax=1, cmap='RdBu_r')
    rasterAx.axvline(no_bins // 2 - .5, c='k', ls=':')
    rasterAx.axis('off')
    if action == 'pL2C':
        rasterAx.axis('on')
        sns.despine(left=True, bottom=True, ax=rasterAx)
        rasterAx.set_xticks(())
        rasterAx.set_yticks((0, len(df)-1))
        rasterAx.set_yticklabels([1,len(df)])
        rasterAx.set_ylabel('neuron', labelpad=-15)
        rasterAx.tick_params(axis='y', length=0, pad=2)
    if action in ['pC2L','pC2R']:
        rasterAx.axis('on')
        sns.despine(left=True, ax=rasterAx, offset=1)
        rasterAx.set_yticks(())
        rasterAx.set_xticks((-.5, no_bins // 2 - .5, no_bins - .5))
        rasterAx.set_xticklabels((-no_bins // 2 * 50, 0, no_bins // 2 * 50))
    
cax = layout.axes['raster_colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='vertical')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(.38, 1.032, '1', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.25, -.05, '-1', ha='center', va='top', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(3.45, .5, 'peak-normalized\naverage', ha='center', va='center', fontdict={'fontsize':7},
         rotation=90, transform=cax.transAxes)
        
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "2choiceIntro.svg")