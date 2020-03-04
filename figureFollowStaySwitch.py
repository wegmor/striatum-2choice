#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:15:03 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FixedLocator
import pathlib
import itertools
import figurefirst
import style
import analysisStaySwitchDecoding
import analysisForcedAlternation
import cmocean
from utils import readSessions, fancyViz, sessionBarPlot
plt.ioff()


#%%
style.set_context()

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
layout = figurefirst.FigureLayout(templateFolder / "followStaySwitch.svg")
layout.make_mplfigures()
    
cachedDataPath = cacheFolder / 'staySwitchAUC.pkl'
if cachedDataPath.is_file():
    staySwitchAUC = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath,
                                                                  n_shuffles=1000)
    staySwitchAUC.to_pickle(cachedDataPath) 

#%% example population rasters etc.
action = 'mR2C'
prevTuned = analysisForcedAlternation.findPrevStaySwitchTuned(
                                        endoDataPath,
                                        alignmentDataPath,
                                        staySwitchAUC)
tasks = pd.read_hdf(endoDataPath, "meta").set_index(["genotype","animal","date"]).task
combinations = itertools.product(("2choice", "forcedAlternation"),
                                 ('stay','switch'),
                                 (1, 2))
for task, stsw, nPrev in combinations:
    if stsw == 'stay':
        aucs = (prevTuned.prevStayTunings.unstack()[action])
    else:
        aucs = (prevTuned.prevSwitchTunings.unstack()[action])
    aucs = aucs[aucs>=nPrev].reset_index()
    aucs = aucs.join(tasks, on=["genotype", "animal", "date"])
    aucs = aucs[aucs.task==task]
    psuffix = "_2prev" if nPrev == 2 else ""
    suffix = stsw+"_followed_"+task+psuffix
    
    stacked = analysisStaySwitchDecoding.getStSwRasterData(endoDataPath, aucs,
                                                           action, task=task,
                                                           pkl_suffix=suffix)
    
    # add whitespace separator between genotypes -- total hack
    sep_rows = stacked.shape[0] // 70
    sep_cols = stacked.shape[1]+4
    sep = pd.DataFrame(np.array([np.nan]*(sep_cols*sep_rows)).reshape((sep_rows,sep_cols)))
    sep.set_index([0,1,2,3], inplace=True)
    sep.index.names = stacked.index.names
    sep.columns = stacked.columns
    stacked = pd.concat([stacked.loc[[gt]].append(sep) for gt in ('d1','a2a')] +
                        [stacked.loc[['oprm1']]])
        
    ccax = layout.axes['{}pop_colorcode_{}'.format(stsw, task)+psuffix]['axis']
    pal = [style.getColor(gt) for gt in ['d1','a2a','oprm1']]
    colorcode = (stacked.reset_index().genotype
                        .replace({'d1':0,'a2a':1,'oprm1':2}).values
                        .reshape(len(stacked),1))
    ccax.imshow(colorcode, cmap=mpl.colors.ListedColormap(pal),
                aspect='auto', interpolation='nearest')
    ccax.axis('off')
    
    trialTypes = ('o!', 'o.', 'r.')
    if task=="forcedAlternation": trialTypes += ('r!',)
    raxs = [layout.axes['{}pop_raster_{}_{}'.format(stsw, tt, task)+psuffix]['axis'] for tt in trialTypes]
    aaxs = [layout.axes['{}pop_avg_{}_{}'.format(stsw, tt, task)+psuffix]['axis'] for tt in trialTypes]
    for i, p in enumerate([action+tt for tt in trialTypes]):
        for gt, gdata in stacked[p].groupby('genotype'):
            m = gdata.mean(axis=0)
            sem = gdata.sem(axis=0)
            aaxs[i].fill_between(np.arange(15), m-sem, m+sem, alpha=0.35, lw=0,
                                 color=style.getColor(gt), clip_on=False,
                                 zorder={'d1':3,'a2a':1,'oprm1':2}[gt])
            aaxs[i].plot(m.values, color=style.getColor(gt), clip_on=False,
                         lw=.5, alpha=.75, zorder={'d1':3,'a2a':1,'oprm1':2}[gt])
        aaxs[i].set_ylim(0, .5)
        aaxs[i].set_xlim(-.5, 14.5)
        aaxs[i].hlines([0], -.5, 16, lw=mpl.rcParams['axes.linewidth'],
                       color='k', alpha=.5, zorder=-99, linestyle=':', clip_on=False)
        aaxs[i].vlines([4.5,9.5], -.18, .5, linestyle=':', color='k', alpha=1, 
                       lw=mpl.rcParams['axes.linewidth'], clip_on=False,
                       zorder=-99)
        aaxs[i].axis('off')

        img = raxs[i].imshow(stacked[p], aspect="auto", interpolation="nearest",
                             vmin=-.5, vmax=.5, cmap="RdYlBu_r")
        raxs[i].axvline(4.5, ls=':', color='k', alpha=1, 
                       lw=mpl.rcParams['axes.linewidth'])
        raxs[i].axvline(9.5, ls=':', color='k', alpha=1,
                        lw=mpl.rcParams['axes.linewidth'])
        raxs[i].axis('off')
        
    aaxs[0].axis('on')
    sns.despine(ax=aaxs[0], trim=True, left=False, right=True, bottom=True,
                offset=1)
    aaxs[0].set_ylabel("z-score")
    aaxs[0].set_yticks((0,.5))
    aaxs[0].set_yticks((.25,), minor=True)
    #aaxs[2].yaxis.set_label_position('right')
    #aaxs[2].yaxis.set_ticks_position('right')
    aaxs[0].set_xticks(())
    
    raxs[0].axis('on')
    sns.despine(ax=raxs[0], left=True, bottom=True)
#    raxs[2].set_ylabel("neuron", labelpad=-5)
#    raxs[2].set_yticks((0, len(stacked)-1))
#    raxs[2].set_yticklabels([1,len(aucs)])
    ylims = raxs[0].get_ylim()
    gtorder = ['d1','a2a','oprm1']
    yticks = (stacked.groupby('genotype').size().loc[gtorder[:-1]].cumsum().values + 
              [sep_rows,2*sep_rows])
    yticks = (np.array([0] + list(yticks)) +
              stacked.groupby('genotype').size().loc[gtorder].values // 2)
    raxs[0].set_yticks(yticks-1)
    raxs[0].set_yticklabels(['{}\n({})'.format(gt,n) for (n,gt) in 
                                 zip(stacked.groupby('genotype').size().loc[gtorder].values,
                                     ['D1','A2A','Oprm1'])],
                            rotation=0, va='center', ha='center')
    raxs[0].set_ylim(ylims) # somehow not having a 0 tick crops the image!
    raxs[0].tick_params(axis='y', length=0, pad=13)
    [tick.set_color(style.getColor(gt)) for (tick,gt) in zip(raxs[0].get_yticklabels(),
                                                             gtorder)]
    raxs[0].set_xticks(())

    cax = layout.axes['{}pop_colorbar_{}'.format(stsw, task)+psuffix]['axis']
    cb = plt.colorbar(img, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cax.set_axis_off()
    cax.text(-.05, .3, -.5, ha='right', va='center', fontdict={'fontsize':6},
             transform=cax.transAxes)
    cax.text(1.05, .3, .5, ha='left', va='center', fontdict={'fontsize':6},
             transform=cax.transAxes)
    cax.text(0.5, -.1, 'z-score', ha='center', va='top', fontdict={'fontsize':6},
             transform=cax.transAxes)

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "followStaySwitch_{}.svg".format(action))
