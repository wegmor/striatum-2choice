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
import analysisTunings
from scipy.spatial.distance import squareform, pdist
from collections import defaultdict
from utils import readSessions
import cmocean
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

tuningData = analysisTunings.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005
    
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
    for action in ['pC2L','pC2R']: #auc.action.unique():
        # select win-stay and lose-switch tuned neurons
        wstTuned = auc.query('action == @action & pct > .995')['neuron'].values
        lswTuned = auc.query('action == @action & auc < .005')['neuron'].values
        
        if len(wstTuned) == 0 and len(lswTuned) == 0:
            print(str(s)+': no strongly stay-switch tuned neurons in '+action+'!')
            continue
        
        selected_labels = [a+tt for a in ('pC2L','pC2R') for tt in ['r.','o.','o!']] #[action+tt for tt in ['r.','o.','o!']]          
        
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
for action, df in (wstLswActionMeans#.loc[wstLswActionMeans.label.str.endswith('o.')]
                                            #.groupby(['action','label'])):
                                            #.groupby([wstLswActionMeans.action,
                                            #          wstLswActionMeans.label.str.slice(0,4)])):
                                    .groupby('action')):
    if not action.startswith('p'): continue
    df = df.copy()
#    if action == 'pC2R':
#        df['value'] *= -1
    df['bin'] = pd.qcut(df.value, 9).cat.codes # every individual action's value is included X # neuron!
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
        #ax = layout.axes['{}_av_x_d'.format(gt)]
        
        for tuning, tdata in gdata.groupby('tuning'):
            ax = layout.axes['{}_av_x_d_{}'.format(gt, 'r' if action=='pC2R' else 'l')]
#            if action == 'pC2R':
            ax.errorbar(tdata['value','mean'], tdata['actionMean','mean'],
                        xerr=tdata['value','sem'], yerr=tdata['actionMean','sem'],
                        #xerr=0, yerr=0,
                        color=style.getColor(tuning), clip_on=False,
                        marker='>' if 'R' in action else '<', markersize=3.3,
                        linestyle='-', markeredgewidth=0) #markerfacecolor='none', 
                        #alpha=1 if label == action else .35)
                        #alpha=1 if action == 'pC2L' else .45)
            
#            if action == 'pC2L':
#                ax.fill_between(tdata['value','mean'],
#                                tdata['actionMean','mean']-tdata['actionMean','sem'],
#                                tdata['actionMean','mean']+tdata['actionMean','sem'],
#                                alpha=.35, color=style.getColor(tuning), lw=0)
#                ax.plot(tdata['value','mean'], tdata['actionMean','mean'], lw=.5,
#                        alpha=.5, color=style.getColor(tuning))
            
#            for _, sdata in session_df.loc[(gt,tuning)].groupby(['animal','date']):
#                ax.plot(sdata.value, sdata.actionMean, clip_on=False,
#                        color=style.getColor(tuning), alpha=.2,
#                        lw=.35)

        ax.axhline(0, lw=mpl.rcParams['axes.linewidth'], color='k', ls=':',
                   alpha=.5, zorder=-1)
        ax.axvline(0, lw=mpl.rcParams['axes.linewidth'], color='k', ls=':',
                   alpha=.5, zorder=-1)

        ax.set_xticks((-5,0,5))
        #ax.set_xticks((3,0,-3))
        ax.set_xticks((2.5,-2.5), minor=True)
        #ax.set_xticks((1.5,-1.5), minor=True)
        ax.set_xlim((-5,5))
        #ax.set_xlim((3,-3))
        ax.set_yticks((-.3,0,.3))
        #ax.set_yticks((-.25,0,.25))
        ax.set_yticks((-.15,.15), minor=True)
        #ax.set_yticks((-.125,.125), minor=True)
        ax.set_ylim((-.3,.3))
        #ax.set_ylim((-.25,.25))
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


#%% pie charts
#layout = figurefirst.FigureLayout(templateFolder / "staySwitchActivity.svg")
#layout.make_mplfigures()

tuning = tuningData.copy()
tuning = tuning.set_index(['genotype','animal','date','neuron'])

for action, auc in staySwitchAUC.query('action in ["pC2L","pC2R"]').groupby('action'):
    auc = auc.set_index('neuron', append=True).copy()
    for staySwitchTuning in ['stayTuned','switchTuned']:
        df = tuning.loc[auc.loc[auc[staySwitchTuning]].index].copy()
        df = df.set_index('action', append=True)
        
        # only keep max tuning for each neuron
        maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
        maxdf.reset_index('action', inplace=True)
        maxdf.loc[~maxdf.signp, 'action'] = 'none' # don't color if not significant
        maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts
        
        # create dictionary with modified alpha to separate r/o/d phases
        cdict = defaultdict(lambda: np.array([1,1,1]),
                            {a:style.getColor(a[:4]) for a 
                             in ['mC2L-','mC2R-','mL2C-','mR2C-','pC2L-','pC2R-','pL2C-','pR2C-']})
        cdict['pL2Cr'] = cdict['pL2C-']
        cdict['pL2Co'] = np.append(cdict['pL2C-'], .45)
        cdict['pL2Cd'] = np.append(cdict['pL2C-'], .7)
        cdict['pR2Cr'] = cdict['pR2C-']
        cdict['pR2Co'] = np.append(cdict['pR2C-'], .45)
        cdict['pR2Cd'] = np.append(cdict['pR2C-'], .7)
        cdict['pC2L-'] = np.append(cdict['pC2L-'], .45)
        
        fig, axs = plt.subplots(3,1)
        for g in ['d1','a2a','oprm1']:
            ax = (layout.axes['pie_{}_{}_{}'.format(g, 'left' if 'L' in action else 'right',
                                                    'stay' if staySwitchTuning == 'stayTuned' else 'switch')]
                             ['axis'])
        
            gdata = maxdf.loc[g]   
            ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                            explode=[.1]*len(gdata),
                            textprops={'color':'k'}, colors=[cdict[a] for a in gdata.index])
            
            ax.set_aspect('equal')


#%%           
def getPercentile(value, shuffle_dist):
    return np.searchsorted(np.sort(shuffle_dist), value) / len(shuffle_dist)

for genotype, auc in staySwitchAUC.query('action in ["pC2L","pC2R"]').groupby('genotype'):
    auc = auc.set_index(['neuron','action'], append=True)[['stayTuned','switchTuned']].unstack()
    # jaccard similarity, neurons pooled
    similarity = pd.DataFrame(1-squareform(pdist(auc.T, metric='jaccard')),
                              index=auc.columns, columns=auc.columns)
    
    similarity_shuffled = []
    for _ in range(1000):
        auc_shuffled = (auc.groupby(['genotype','animal','date'])
                           .apply(lambda g: g.apply(np.random.permutation)))
        similarity_shuffled.append(1-squareform(pdist(auc_shuffled.T, metric='jaccard')))
    
    #similarity_shuffled = np.array(similarity_shuffled)
    #similarity_shuffled = np.apply_along_axis(np.sort, 0, similarity_shuffled)
    similarity_shuffled = pd.concat([pd.DataFrame(s, index=similarity.index, columns=similarity.columns) 
                                         for s in similarity_shuffled],
                                     axis=1, keys=np.arange(len(similarity_shuffled)))
    
    similarity = similarity.stack(level=(0,1))
    similarity_shuffled = similarity_shuffled.stack(level=(1,2))
    percentile = similarity_shuffled.apply(lambda g: getPercentile(similarity.loc[g.name], g),
                                           axis=1)
    similarity = similarity.unstack(level=(-2,-1))
    percentile = percentile.unstack(level=(-2,-1))
    similarity_shuffled = similarity_shuffled.unstack(level=(-2,-1))
    similarity_shuffled.columns = similarity_shuffled.columns.reorder_levels((1,2,0))


    leftStay, leftSwitch = ('stayTuned','pC2L'), ('switchTuned','pC2L')
    rightStay, rightSwitch = ('stayTuned','pC2R'), ('switchTuned','pC2R')
    
    stayXstay = (leftStay, rightStay)
    switchXswitch = (leftSwitch, rightSwitch)
    lstayXrswitch = (leftStay, rightSwitch)
    lswitchXrstay = (leftSwitch, rightStay)
    comps = [stayXstay, switchXswitch, lstayXrswitch, lswitchXrstay]
    
    bp_data = np.stack([similarity_shuffled.loc[comp].values for comp in comps],
                       axis=1)
    observed = np.array([similarity.loc[comp] for comp in comps])
        
    ax = layout.axes['{}_overlap'.format(genotype)]['axis']
    sns.boxplot(data=bp_data, ax=ax, color=style.getColor('shuffled'),
                showcaps=False, showfliers=False,
                boxprops={'alpha':0.25, 'lw':0, 'zorder':-99, 'clip_on':False}, 
                width=.6, whiskerprops={'c':'k','zorder':99, 'clip_on':False},
                medianprops={'c':'k','zorder':99, 'clip_on':False})
    ax.scatter(np.arange(4), observed, color='k', marker='x', clip_on=False)
    
    y = -.025
    ax.scatter(-.2,y, marker='<', color=style.getColor('stay'), clip_on=False)
    ax.scatter(.2,y, marker='>', color=style.getColor('stay'), clip_on=False)
    ax.scatter(.8,y, marker='<', color=style.getColor('switch'), clip_on=False)
    ax.scatter(1.2,y, marker='>', color=style.getColor('switch'), clip_on=False)
    ax.scatter(1.8,y, marker='<', color=style.getColor('stay'), clip_on=False)
    ax.scatter(2.2,y, marker='>', color=style.getColor('switch'), clip_on=False)
    ax.scatter(2.8,y, marker='<', color=style.getColor('switch'), clip_on=False)
    ax.scatter(3.2,y, marker='>', color=style.getColor('stay'), clip_on=False)
    #[ax.text(x, y, '&', ha='center', va='center') for x in range(4)]
    
    ax.set_ylim((0,.2))
    ax.set_yticks((0,.1,.2))
    ax.set_yticks((.05,.15), minor=True)
    if genotype == 'd1':
        ax.set_ylabel('% overlap')
        ax.set_yticklabels((0,10,20))
    else:
        ax.set_yticklabels(())
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(())
    sns.despine(ax=ax)
    

#%%
def getEventWindows(events, win_size=(20, 19)):
    windows = pd.DataFrame()
    for s in readSessions.findSessions(endoDataPath, task='2choice'):
        lfa = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
        deconv = s.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
        
        if not len(lfa) == len(deconv):
            print(str(s)+': more labeled frames than signal!')
            continue
    
        actionLabels = lfa.groupby('actionNo').label.first()
        av = actionValues.loc[(s.meta.genotype,s.meta.animal,s.meta.date)]
        
        if not (av['label'] == actionLabels).all():
            print(str(s)+': labeling does not match!')
            continue
    
        lfa['index'] = deconv.index
        deconv = deconv.set_index([lfa.label,lfa.actionNo], append=True)
        deconv.index.names = ['index','label','actionNo']
        deconv.columns.name = 'neuron'
        
        center_idx = (lfa.loc[lfa.label.str.slice(0,4).isin(events)].groupby('actionNo')
                                 [['index','actionNo','label']].first().values)
    
        _windows = []
        neurons = deconv.columns
        for idx, actionNo, label in center_idx:
            win = deconv.loc[idx-win_size[0]:idx+win_size[1]].reset_index()
            win.loc[win.actionNo > actionNo, neurons] = np.nan
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
    
    return windows
    
#%%
#windows = getEventWindows(['mR2C','mL2C'])
#windows.to_pickle('mrlEventWindows.pkl')
    
#%%
av = actionValues.copy()
auc = staySwitchAUC.copy()
auc = auc.reset_index().set_index(['action','genotype','animal','date','neuron'])
tuning = tuningData.copy()
tuning = tuning.set_index(['genotype','animal','date','neuron','action'])
tuning = tuning.loc[tuning.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
tuning = tuning.reset_index('action')
tuning.loc[~tuning.signp, 'action'] = 'none'
windows = pd.read_pickle('mrlEventWindows.pkl')

windows.set_index(['genotype','animal','date','actionNo'], inplace=True)
windows['value'] = av.value
windows.reset_index(inplace=True)
windows['action'] = windows.label.str.slice(0,4)
windows.set_index(['genotype','animal','date','neuron'], inplace=True)
#windows['stay'] = (auc.pct > .995).astype('int') - (auc.pct < .005).astype('int')
windows['mR2C_stay'] = auc.loc['mR2C', 'pct'] > .995
windows['mL2C_stay'] = auc.loc['mL2C', 'pct'] > .995
windows['mR2C_auc'] = auc.loc['mR2C', 'auc']
windows['mL2C_auc'] = auc.loc['mL2C', 'auc']
windows['auc'] = windows[['mL2C_auc','mR2C_auc']].max(axis=1)
#windows.reset_index('action', inplace=True)
windows['trial_mean'] = windows['frameNo'].mean(axis=1)
windows['pTuning'] = tuning.action


#%%
#cmap = sns.color_palette('hsv', 7)
cmap = [cmocean.cm.phase(c) for c in np.linspace(.12,.88,7)]
for side in ['pC2R','pC2L']:
    for genotype, df in (windows.loc[windows['{}_stay'.format(side)] &
                                     windows.label.str.endswith('.') &
                                     (windows.action == side) &
                                     (windows.pTuning == side+'-')]
                                .groupby('genotype')):
#        df = df.copy()
#        bins = [-10, -2.5, -1.25, -.5, .5, 1.25, 2.5, 10]
#        df['bin'] = pd.cut(df.value, bins)
        if 'R' in side:
            df.value *= -1
        #df['bin'] = pd.qcut(df.value, 4)
        bins = [-10,1,3,10]
        df['bin'] = pd.cut(df.value, bins)
        
        ax = layout.axes['{}_pC_{}stay'.format(genotype, 'r' if 'R' in side else 'l')]['axis']

        for c,(b, bdf) in enumerate(df.groupby('bin')):
            neuronAvgs = bdf.frameNo.groupby(['animal','neuron']).mean()
            mean = neuronAvgs.mean()
            sem = neuronAvgs.sem()
            ax.fill_between(np.arange(len(mean)), mean-sem, mean+sem,
                            label=b, color=cmap[c], alpha=.35, lw=0)
            ax.plot(mean, color=sns.desaturate(cmap[c],.75), lw=.7, alpha=.85)
            ax.axvline(20, c='k', ls=':', alpha=.5, lw=mpl.rcParams['axes.linewidth'],
                       zorder=-99)
        
        ax.set_xlim((0,30))
        ax.set_ylim((-.2, 1))
        ax.set_yticks((0,.5,1))
        ax.set_yticks((.25,.75), minor=True)
        ax.set_yticklabels(())
        if genotype == 'd1':
            ax.set_yticklabels(ax.get_yticks())
            ax.set_ylabel('sd')
        ax.set_xticks((0,10,20,30))
        ax.set_xticks((5,15,25), minor=True)
        ax.set_xticklabels(())
        if ('L' in side) & (genotype == 'a2a'):
            ax.set_xticklabels((-1.,-.5,0,.5))
            ax.set_xlabel('time (s)')
        sns.despine(ax=ax)
   
ax = layout.axes['colorbar']['axis']     
cbar = mpl.colors.ListedColormap(cmap)
#cbar.set_under(cmap[0])
#cbar.set_over(cmap[-1])
bounds = [-5]+bins[1:-1]+[5]
norm = mpl.colors.BoundaryNorm(bounds, cbar.N)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cbar, norm=norm, orientation='horizontal',
                               extend='neither', ticks=[-5,-2.5,0,2.5,5],
                               spacing='proportional')
cb.outline.set_visible(False)
ax.tick_params(bottom=False, pad=-.4)
ax.set_xticklabels([-5,-2.5,0,2.5,5], rotation=0)
ax.text(.5, 1.25, 'action value', ha='center')


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchActivity.svg")


#%%
df = (windows.loc[windows.mL2C_stay | windows.mR2C_stay].reset_index()
             .set_index(['auc','genotype','animal','date','neuron'])
             .sort_index(ascending=False).copy())

#%%
for neuron, data in (df.groupby(['auc', 'genotype','animal','date','neuron'],
                                sort=False)):
    print(data.reset_index().auc.unique())
    data = data.loc[data.label.str.slice(-2).isin(['r.','o.','o!'])].copy()
    
    fig, (lax, rax) = plt.subplots(1, 2, sharey=True)

    for label, sdata in data.groupby('label'):
        ax = lax if 'L' in label else rax
        sns.regplot('value', 'trial_mean', data=sdata, ax=ax, fit_reg=True, ci=0,
                    color=style.getColor(label[-2:]), scatter_kws={'alpha':.5})
        
    for action, sdata in data.groupby('action'):
        ax = lax if 'L' in action else rax
        sns.regplot('value', 'trial_mean', data=sdata, ax=ax, ci=68,
                    x_bins=4, color='k')
        
    lax.set_xlim((6,-1))
    lax.set_ylim((-.5,1))
    rax.set_xlim((1,-6))
    rax.set_ylim((-.5,1))
    
    plt.show()
