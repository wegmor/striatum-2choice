#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:37:51 2019

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
from utils import readSessions
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
cachedDataPath = cacheFolder / 'staySwitchAUC.pkl'
if cachedDataPath.is_file():
    staySwitchAUC = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath)
    staySwitchAUC.to_pickle(cachedDataPath)
    
cachedDataPaths = [cacheFolder / name for name in ['actionValues.pkl',
                                                   'logRegCoefficients.pkl',
                                                   'logRegDF.pkl']]
if np.all([path.is_file() for path in cachedDataPaths]):
    actionValues = pd.read_pickle(cachedDataPaths[0])
    logRegCoef = pd.read_pickle(cachedDataPaths[1])
    logRegDF = pd.read_pickle(cachedDataPaths[2])
else:
    actionValues, logRegCoef, logRegDF = analysisStaySwitchDecoding.getActionValues(endoDataPath)
    actionValues.to_pickle(cachedDataPaths[0])
    logRegCoef.to_pickle(cachedDataPaths[1])
    logRegDF.to_pickle(cachedDataPaths[2])
    
    
#%%
actionValues.set_index(['genotype','animal','date','actionNo'], inplace=True)
actionValues.sort_index(inplace=True)
staySwitchAUC.set_index(['genotype','animal','date'], inplace=True)
staySwitchAUC.sort_index(inplace=True)

#%%
wstLswActionMeans = pd.DataFrame()
for s in readSessions.findSessions(endoDataPath, task='2choice'):
    lfa = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
    deconv = s.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
    
    if not len(lfa) == len(deconv):
        print(str(s)+': more labeled frames than signal!')
        continue
    
    auc = staySwitchAUC.loc[(s.meta.genotype,s.meta.animal,s.meta.date)]
    auc = auc.query('pct > .995 | pct < .005').copy()
    
    if auc.empty:
        print(str(s)+': no strongly stay-switch tuned neurons AT ALL!')
        continue
    
    # compute mean of every action for tuned neurons
    actionLabels = lfa.groupby('actionNo').label.first()
    actionMeans = (deconv[auc.neuron.unique()]
                         .groupby(lfa.actionNo).mean())
    av = actionValues.loc[(s.meta.genotype,s.meta.animal,s.meta.date)]
    
    if not (av['label'] == actionLabels).all():
        print(str(s)+': labeling does not match!')
        continue
    
    # loop over actions for which tunings have been computed
    for action in auc.action.unique():
        # select win-stay and lose-switch tuned neurons
        wstTuned = auc.query('action == @action & pct > .995')['neuron'].values
        lswTuned = auc.query('action == @action & auc < .005')['neuron'].values
        
        if len(wstTuned) == 0 and len(lswTuned) == 0:
            print(str(s)+': no strongly stay-switch tuned neurons in '+action+'!')
            continue
        
        selected_labels = [action+tt for tt in ['r.','o.','o!']]          
        
        # select action means for reward-stay, omission-stay and omission switch trials
        # for tuned neurons
        am = actionMeans.loc[actionLabels.isin(selected_labels)]
        wst_am = am[wstTuned].copy()
        lsw_am = am[lswTuned].copy()
        am = pd.concat([wst_am, lsw_am], keys=['stay','switch'],
                       axis=1)
        am = pd.DataFrame(am.stack((0,1)), columns=['actionMean'])
        am.index.names = ['actionNo','tuning','neuron']
        
        # add action values and session meta data to session/action data frame
        am = pd.merge(am, av, left_index=True, right_index=True).dropna().reset_index()
        for (k,v) in [('action',action),('date',s.meta.date),
                      ('animal',s.meta.animal),('genotype',s.meta.genotype)]:
            am.insert(0,k,v)
        
        # append session/action data to store
        wstLswActionMeans = wstLswActionMeans.append(am, ignore_index=True)
        
                
#%%
layout = figurefirst.FigureLayout(templateFolder / "staySwitchActivity.svg")
layout.make_mplfigures()

#%%
#for action, df in wstLswActionMeans.groupby('action'):
for action, df in wstLswActionMeans.loc[wstLswActionMeans.label.str.endswith('o.')].groupby('action'):
    if not action.startswith('p'): continue
    df = df.copy()
    df['bin'] = pd.qcut(df.value, 4).cat.codes # every individual action's value is included X # neuron!
    # v: binned average for tuned populations / session
    session_df = df.groupby(['genotype','tuning','animal','date','bin'])[['value','actionMean']].mean()
    session_df['noNeurons'] = df.groupby(['genotype','tuning','animal','date']).neuron.nunique()
    # v: across sessions
#    avg_df = session_df.groupby(['genotype','tuning','bin'])[['value','actionMean']].agg(['mean','sem'])
    grouping = session_df.groupby(['genotype','tuning','bin'])
    value_wAvg = grouping.apply(analysisStaySwitchDecoding.wAvg, 'value', 'noNeurons')
    value_sem = grouping.apply(analysisStaySwitchDecoding.bootstrap, 'value', 'noNeurons')
    mean_wAvg = grouping.apply(analysisStaySwitchDecoding.wAvg, 'actionMean', 'noNeurons')
    mean_sem = grouping.apply(analysisStaySwitchDecoding.bootstrap, 'actionMean', 'noNeurons')
    value = pd.concat([value_wAvg, value_sem], axis=1, keys=['mean','sem'])
    mean = pd.concat([mean_wAvg, mean_sem], axis=1, keys=['mean','sem'])
    avg_df = pd.concat([value, mean], axis=1, keys=['value','actionMean'])

    for gt, gdata in avg_df.groupby('genotype'):
        ax = layout.axes['{}_av_x_d'.format(gt)]
        
        for tuning, tdata in gdata.groupby('tuning'):
            ax.errorbar(tdata['value','mean'], tdata['actionMean','mean'],
                        xerr=tdata['value','sem'], yerr=tdata['actionMean','sem'],
                        color=style.getColor(tuning), clip_on=False,
                        marker='>' if 'R' in action else '<')
            
#            for _, sdata in session_df.loc[(gt,tuning)].groupby(['animal','date']):
#                ax.plot(sdata.value, sdata.actionMean, clip_on=False,
#                        color=style.getColor(tuning), alpha=.2,
#                        lw=.35)

        ax.axhline(0, lw=mpl.rcParams['axes.linewidth'], color='k', ls=':',
                   alpha=.5, zorder=-1)
        ax.axvline(0, lw=mpl.rcParams['axes.linewidth'], color='k', ls=':',
                   alpha=.5, zorder=-1)

        #ax.set_xticks((5,0,-5))
        ax.set_xticks((3,0,-3))
        #ax.set_xticks((2.5,-2.5), minor=True)
        ax.set_xticks((1.5,-1.5), minor=True)
        #ax.set_xlim((5,-5))
        ax.set_xlim((3,-3))
        #ax.set_yticks((-.4,0,.4))
        ax.set_yticks((-.3,0,.3))
        #ax.set_yticks((-.2,.2), minor=True)
        ax.set_yticks((-.15,.15), minor=True)
        #ax.set_ylim((-.4,.4))
        ax.set_ylim((-.3,.3))
        if gt != 'd1':
            ax.set_yticklabels(())
        else:
            ax.set_ylabel('sd')
        if gt == 'a2a':
            ax.set_xlabel('action value')
#        ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[gt],
#                     pad=11)
        sns.despine(ax=ax)
     
axt = layout.axes['legend']
legend_elements = [mlines.Line2D([0], [0], marker='<', color='k', #markersize=2.8,
                         markeredgewidth=0, label='(left) choice', lw=0),
                   mpatches.Patch(color=style.getColor('stay'), alpha=1,
                                  label='win-stay tuned'),
                   mpatches.Patch(color=style.getColor('switch'), alpha=1,
                                  label='lose-switch tuned'),
                  ]
axt.legend(handles=legend_elements, ncol=len(legend_elements), loc='center',
           mode='expand')
axt.axis('off')


#%%
staySwitchAUC['stayTuned'] = staySwitchAUC.pct > .995
staySwitchAUC['switchTuned'] = staySwitchAUC.pct < .005

auc = staySwitchAUC.groupby(['action','genotype','animal','date'])[['stayTuned','switchTuned']].sum().copy()
auc['noNeurons'] = staySwitchAUC.groupby(['action','genotype','animal','date']).size()
auc.loc[:,['stayTuned','switchTuned']] = auc[['stayTuned','switchTuned']] / auc.noNeurons.values[:,np.newaxis]

auc = auc.query('action in ["pC2R","pC2L"]').copy()

#%%
for (genotype, action), df in auc.groupby(['genotype','action']):
    stay_avg = analysisStaySwitchDecoding.wAvg(df, 'stayTuned', 'noNeurons')
    stay_sem = analysisStaySwitchDecoding.bootstrap(df, 'stayTuned', 'noNeurons')
    switch_avg = analysisStaySwitchDecoding.wAvg(df, 'switchTuned', 'noNeurons')
    switch_sem = analysisStaySwitchDecoding.bootstrap(df, 'switchTuned', 'noNeurons')

    ax = layout.axes['{}_pct_{}'.format(genotype, 'left' if 'L' in action else 'right')]
    ax.bar([0,1], [stay_avg,switch_avg], yerr=[stay_sem,switch_sem],
           lw=0, alpha=.5, zorder=1, color=[style.getColor('stay'),style.getColor('switch')])
    for sess in df.itertuples():
        ax.plot([0,1], [sess.stayTuned,sess.switchTuned], color='k', alpha=.2,
                lw=sess.noNeurons/250, clip_on=False)
        
    ax.set_xlim((-.6,1.6))
    ax.set_ylim((0,.4))
    ax.set_yticks((0,.4))
    ax.set_yticks((.2,), minor=True)
    ax.set_xticks(())
    if (genotype == 'd1') and (action == 'pC2L'):
        ax.set_ylabel('% neurons')
        ax.set_yticklabels((0,40))
    else:
        ax.set_yticklabels(())
    if "L" in action:
        ax.scatter(.5, .4, marker='<', color='k', clip_on=False)
    else:
        ax.scatter(.5, .4, marker='>', color='k', clip_on=False)
    sns.despine(ax=ax)


#%%    
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchActivity.svg")
