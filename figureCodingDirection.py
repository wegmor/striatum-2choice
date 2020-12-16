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
from utils import readSessions
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


#%% plot example neuron
incl_actions=['pR2C','mR2C','pC2R','pC2L']
sess = next(readSessions.findSessions(endoDataPath, animal='5308', date='190201'))
aucs = pd.read_pickle('cache/staySwitchAUC.pkl').query('animal == "5308" & date == "190201"')
aucs.set_index('action', inplace=True)
neuron = aucs.iloc[aucs.loc['mR2C','auc'].argmax()].neuron
deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)[[neuron]]
deconv.columns.name = 'neuron'
lfa = sess.labelFrameActions(reward='fullTrial', switch=True).reset_index(drop=True)
lfa['bin'] = lfa.actionProgress // (1/5)
actionMeans = deconv.groupby([lfa['label'],lfa['actionNo'],lfa['bin']]).mean()
acIndex = pd.CategoricalIndex(actionMeans.index.get_level_values(0).str.slice(0,-2), incl_actions, ordered=True, name='action')
ttIndex = pd.Index(actionMeans.index.get_level_values(0).str.slice(-2), name='trialType')
actionMeans = actionMeans.set_index([acIndex,ttIndex], append=True).reset_index('label',drop=True)
actionMeans.index = actionMeans.index.reorder_levels((1,2,3,0))
actionMeans = actionMeans.query('action in @incl_actions & trialType in ["r.","o.","o!"]')
ttMeans = actionMeans.groupby(['trialType','action','bin']).mean()


ax = layout.axes['subMean']['axis']

rst = ttMeans.loc['r.'].loc[['pR2C','mR2C','pC2R']].values
ost = ttMeans.loc['o.'].loc[['pR2C','mR2C','pC2R']].values
osw = ttMeans.loc['o!'].loc[['pR2C','mR2C','pC2L']].values
mean = np.stack([rst,ost,osw]).mean(axis=0)
x = len(mean)

ax.plot(rst, c=style.getColor('r.'))
ax.plot(ost, c=style.getColor('o.'))
ax.plot(osw,c=style.getColor('o!'), markersize=1)
ax.plot(np.arange(x*2,x*3), mean, c=style.getColor('shuffled'))
ax.plot(np.arange(x*4,x*5), rst-mean, c=style.getColor('r.'))
ax.plot(np.arange(x*4,x*5), ost-mean, c=style.getColor('o.'))
ax.plot(np.arange(x*4,x*5), osw-mean, c=style.getColor('o!'))
ax.hlines([0,0,0], [-.2*x,1.8*x,3.8*x], [1.2*x,3.2*x,5.2*x], ls=':', 
          color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
ax.vlines(np.array([1/3,2/3,2+1/3,2+2/3,4+1/3,4+2/3])*x-.5, -.35, .85, ls=':', color='k',
          lw=mpl.rcParams['axes.linewidth'], zorder=-99)
ax.text(x*1.5, .225, '−', ha='center', va='center', fontsize=10)
ax.text(x*3.5, .225, '=', ha='center', va='center', fontsize=10)

ax.set_ylabel('z-score')
ax.set_ylim((-.35,.85))
ax.set_xticks(())
ax.set_xlim((-.2*x,5.2*x))
ax.set_yticks((0,.5))
ax.set_yticks((0.25,), minor=True)
sns.despine(trim=True, ax=ax, bottom=True, offset=.1)

legend_elements = [mpl.lines.Line2D([],[], color=style.getColor('r.'), alpha=1,
                                    label='win-stay'),                          
                   mpl.lines.Line2D([],[], color=style.getColor('o.'), alpha=1,
                                    label='lose-stay'),
                   mpl.lines.Line2D([], [], color=style.getColor('o!'), alpha=1,
                                    label='lose-switch'),
                  ]
ax.legend(handles=legend_elements, ncol=len(legend_elements), loc='upper center',
          bbox_to_anchor=(.4,-.1), columnspacing=1)


#%% load coding direction data (traces and means)
cdMeans = analysisStaySwitchDecodingSupp.getStSwCodingDirectionRaster(endoDataPath)

cmap = (mpl.colors.LinearSegmentedColormap
           .from_list('wStLSw', [style.getColor('o!'), (1,1,1), style.getColor('r.')], 256))
cbar = mpl.colors.LinearSegmentedColormap.from_list('test',[style.getColor(c) for c in cdMeans.columns],
                                                    len(cdMeans.columns))

cdMeansBinned, cdMeansBinnedRS = analysisStaySwitchDecodingSupp.getStSwCodingDirectionTraces(endoDataPath)
cdSEMs = (pd.concat(cdMeansBinnedRS, keys=np.arange(len(cdMeansBinnedRS)), names=['sampleNo'])
            .groupby(['genotype','trialType','action','bin']).std())

#%%
sem = (pd.concat(cdMeansBinnedRS, keys=np.arange(len(cdMeansBinnedRS)), names=['sampleNo'])
         .groupby(['genotype','trialType','action','bin']).std()).loc['oprm1','r.']
rst = cdMeansBinned.loc['oprm1','r.']


#%% plot coding direction trace examples
leftTrial = ['pL2C','mL2C','pC2L','mC2L','dL2C']
rightTrial = ['pR2C','mR2C','pC2R','mC2R','dR2C']
ax = layout.axes['CDTraces']['axis']
for tuning in rst.columns:
    offset = -.225 if tuning in rightTrial else .025
    ax.plot(np.arange(len(rst)), rst[tuning].values+offset, color=style.getColor(tuning),
            clip_on=False, lw=1, zorder=-99)
    ax.fill_between(np.arange(len(rst)),
                    rst[tuning].values+sem[tuning].values+offset,
                    rst[tuning].values-sem[tuning].values+offset,
                    color=style.getColor(tuning), alpha=.35, zorder=-99)
ax.hlines([-.25, 0, .25], -6, 50, ls='-', color='k', clip_on=False)
ax.vlines([-.5,24.5,49.5], -.25, .25, ls='-', color='k', clip_on=False)
ax.vlines(np.arange(-.5,50.5,5), -.25, .25, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          clip_on=False)
ax.set_ylim((-.25,.25))
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
rightStay = ['pR2C','mR2C','pC2R','mC2R','dR2C']
rightSwitch = ['pR2C','mR2C','pC2L','mC2L','dL2C']
rst = cdMeansBinned.loc['oprm1','r.'].loc[rightStay]
rstSem = cdSEMs.loc['oprm1','r.'].loc[rst.index]
ost = cdMeansBinned.loc['oprm1','o.'].loc[rightStay]
ostSem = cdSEMs.loc['oprm1','o.'].loc[ost.index]
osw = cdMeansBinned.loc['oprm1','o!'].loc[rightSwitch]
oswIndex = pd.CategoricalIndex(osw.index.get_level_values(0), rightSwitch, ordered=True)
osw = osw.set_index([oswIndex, osw.index.get_level_values(1)]).sort_index()
oswSem = cdSEMs.loc['oprm1','o!'].loc[osw.index]

ax = layout.axes['mR2C_cd']['axis']

ax.plot(rst['mR2C'].values, color=style.getColor('r.'))
ax.fill_between(np.arange(len(rst)), 
                rst['mR2C'].values+rstSem['mR2C'].values,
                rst['mR2C'].values-rstSem['mR2C'].values,
                color=style.getColor('r.'), alpha=.35, lw=0)
ax.plot(ost['mR2C'].values, color=style.getColor('o.'))
ax.fill_between(np.arange(len(ost)), 
                ost['mR2C'].values+ostSem['mR2C'].values,
                ost['mR2C'].values-ostSem['mR2C'].values,
                color=style.getColor('o.'), alpha=.35, lw=0)
ax.plot(osw['mR2C'].values, color=style.getColor('o!'))
ax.fill_between(np.arange(len(osw)), 
                osw['mR2C'].values+oswSem['mR2C'].values,
                osw['mR2C'].values-oswSem['mR2C'].values,
                color=style.getColor('o!'), alpha=.35, lw=0)

trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.vlines(np.arange(5,25,5)-.5, 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          transform=trans)
ax.axhline(0, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'], alpha=.5,
           zorder=-99)
ax.set_yticks((-.1,0,.1))
ax.set_ylim((-.2,.2))
ax.set_xticks(())
ax.set_ylabel('coding direction\n(z-score)')
sns.despine(ax=ax, bottom=True, trim=True)


#%%
ax = layout.axes['leftRewardStayCD']['axis']

leftStay = ['pL2C','mL2C','pC2L','mC2L','dL2C']
rst = cdMeansBinned.loc['oprm1','r.'].loc[leftStay]
rstSem = cdSEMs.loc['oprm1','r.'].loc[rst.index]

offsets = -np.arange(len(leftStay))*.1
transY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
transX = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
for p, tuning in enumerate(leftStay):
    offset = offsets[p]
    ax.plot(rst[tuning].values+offset, color=style.getColor(tuning), lw=1.25,
            clip_on=False)
    ax.fill_between(np.arange(len(rst)), 
                    rst[tuning].values+rstSem[tuning].values+offset,
                    rst[tuning].values-rstSem[tuning].values+offset,
                    color=style.getColor(tuning), alpha=.35, lw=0)
    ax.fill_between(np.array([p*5,(p+1)*5])-.5, 0, 1, color=style.getColor(tuning), alpha=.075,
                    zorder=-9, lw=0, transform=transY)
    ax.fill_between(np.array([p*5,(p+1)*5])-.5, 0, -.075, color=style.getColor(tuning), alpha=1,
                    zorder=-9, lw=0, transform=transY, clip_on=False)
ax.vlines(np.arange(5,25,5)-.5, -.075, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          transform=transY)
ax.hlines(offsets, 0, 1, ls=':', color=[style.getColor(t) for t in leftStay],
          lw=mpl.rcParams['axes.linewidth'], alpha=1, transform=transX)
ax.hlines(offsets[:-1]-.05, 0, 1, color='k', lw=mpl.rcParams['axes.linewidth'],
          transform=transX, alpha=1)
ax.set_ylim((-.45,.05))
ax.set_xlim((-.5,24.5))
ax.set_xticks(())
ax.set_yticks(offsets)
ax.set_yticklabels(['left outcome\n(in port)', 'left to center\n(turn)', 'center to left\n(in port)',
                    'center to left\n(turn)', 'left wait\n(in port)'])
[l.set_color(style.getColor(t)) for l,t in zip(ax.get_yticklabels(), leftStay)]
ax.set_xlabel('trial phase', labelpad=8)
ax.set_ylabel('trial phase-specific\ntrial type-coding population')
ax.set_title('Oprm1 phase-specific\ntrial type coding\n(left win-stay trial)',
             pad=10)


#%% plot matrices
cax = layout.axes['CDlegend']['axis']

for gt in ['d1','a2a','oprm1']:
    ax = layout.axes[gt+'CD']['axis']
    sns.heatmap(cdMeans.loc[gt].T, center=0, cmap=cmap, square=True, vmin=-.1, vmax=.1, ax=ax,
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
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)