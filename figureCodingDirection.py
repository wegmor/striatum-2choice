#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:02:22 2020

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
import analysisStaySwitchDecodingSupp

plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()


#%%
svgName = 'codingDirection.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()


#%%
cdMeans = analysisStaySwitchDecodingSupp.getStSwCodingDirectionAvgs(endoDataPath)

cmap = (mpl.colors.LinearSegmentedColormap
           .from_list('wStLSw', [style.getColor('o!'), (1,1,1), style.getColor('r.')], 256))
cbar = mpl.colors.LinearSegmentedColormap.from_list('test',[style.getColor(c) for c in cdMeans.columns],
                                                    len(cdMeans.columns))
cax = layout.axes['CDlegend']['axis']

for gt in ['d1','a2a','oprm1']:
    ax = layout.axes[gt+'CD']['axis']
    sns.heatmap(cdMeans.loc[gt].T, center=0, cmap=cmap, square=True, vmin=-.2, vmax=.2, ax=ax,
                cbar_ax=cax, cbar_kws={'orientation':'horizontal'})
    ax.vlines(np.arange(0,31,5), 0, 10, color='k', clip_on=False)
    ax.vlines(np.arange(0,31), 0, 10, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines(np.arange(1,10), 0, 30, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines([0,5,10], 0, 30, ls='-', color='k', clip_on=False)
    ax.set_yticks(())
    ax.set_xticks(())
    ax.set_ylabel('trial phase-specific\ntrial type-coding population', labelpad=18, fontsize=8)
    ax.set_xlabel('trial phase', labelpad=26, fontsize=8)

cax.set_ylabel('', fontsize=7)
cax.tick_params(axis='x', length=0)
cbarYAx = layout.axes['oprm1CD_ybar']['axis']
cbarXAx = layout.axes['oprm1CD_xbar']['axis']
cbarYAx.pcolormesh((cdMeans.columns.codes / 10)[:,np.newaxis], cmap=cbar)
cbarYAx.hlines(np.arange(1,10), 0, 1, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'])
cbarYAx.hlines([0,5,10], -2, 1, ls='-', color='k', clip_on=False)
cbarYAx.set_xlim((0,1))
cbarXAx.pcolormesh((cdMeans.loc[gt].index.get_level_values(1).codes / 10)[np.newaxis,:], cmap=cbar)
cbarXAx.vlines(np.arange(1,30), 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'])
cbarXAx.vlines([5,15,25], 1, -2, ls='-', color='k', clip_on=False)
cbarXAx.vlines([0,10,20,30], 1, -4, ls='-', color='k', clip_on=False)
cbarXAx.set_ylim((0,1))
cbarYAx.axis('off')
cbarYAx.invert_yaxis()
cbarXAx.axis('off')


#%%
cdMeansBinned = analysisStaySwitchDecodingSupp.getStSwCodingDirectionAvgs(endoDataPath, binPhases=True)
rst = cdMeansBinned.loc['oprm1','r.']

leftTrial = ['pL2C','mL2C','pC2L','mC2L','dL2C']
rightTrial = ['pR2C','mR2C','pC2R','mC2R','dR2C']
ax = layout.axes['CDTraces']['axis']
for tuning in rst.columns:
    offset = -.3 if tuning in rightTrial else .05
    ax.plot(np.arange(len(rst)), rst[tuning].values+offset, color=style.getColor(tuning),
            clip_on=False, lw=1, zorder=-99)
ax.hlines([-.35, 0, .35], -6, 50, ls='-', color='k', clip_on=False)
ax.vlines([-.5,24.5,49.5], -.35, .35, ls='-', color='k', clip_on=False)
ax.vlines(np.arange(-.5,50.5,5), -.35, .35, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          clip_on=False)
ax.set_ylim((-.35,.35))
ax.set_xlim((-.5,49.5))
ax.set_xticks(())
ax.set_yticks(())
ax.set_ylabel('trial phase-specific\ntrial type-coding', labelpad=12, fontsize=8)
ax.set_xlabel('trial phase', labelpad=26, fontsize=8)

cbarXAx = layout.axes['oprm1CD_xbar2']['axis']
cbarXAx.pcolormesh((rst.index.get_level_values(0).codes / 10)[np.newaxis,:], cmap=cbar, lw=0)
cbarXAx.vlines(np.arange(0,50,5), 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'])
cbarXAx.vlines([25,], 1, -2, ls='-', color='k', clip_on=False)
cbarXAx.vlines([0,50], 1, -4, ls='-', color='k', clip_on=False)
cbarXAx.set_ylim((0,1))
cbarXAx.axis('off')


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)