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
from matplotlib.ticker import MultipleLocator
import pathlib
import figurefirst
import style
import analysisStaySwitchDecoding
import cmocean
plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path(".") / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path(".") / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path(__file__).parent / "templates"

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "staySwitchDecoding.svg")
layout.make_mplfigures()

#%%
cachedDataPaths = [cacheFolder / name for name in ['stsw_m.pkl','stsw_p.pkl',
                                                   'stsw_c.pkl']]
if np.all([path.is_file() for path in cachedDataPaths]):
    M = pd.read_pickle(cachedDataPaths[0])
    P = pd.read_pickle(cachedDataPaths[1])
    C = pd.read_pickle(cachedDataPaths[2])
else:
    M = pd.DataFrame() # confusion matrices (shuffle and real)
    P = pd.DataFrame() # action (probability) predictions
    C = pd.DataFrame() # svm coefficients
    
    for action in ['mL2C','pC2L','mC2L','mR2C','pC2R','mC2R']:
        (rm,rp,rc), (sm,sp,sc) = analysisStaySwitchDecoding.decodeStaySwitch(endoDataPath, action)
        
        for df in [rm,rp,rc,sm,sp,sc]:
            df.insert(0, 'action', action)
        
        m = pd.concat([rm,sm], axis=0, keys=[False,True], names=['shuffled']).reset_index('shuffled')
        M = M.append(m, ignore_index=True)
        
        p = pd.concat([rp,sp], axis=0, keys=[False,True], names=['shuffled']).reset_index('shuffled')
        P = P.append(p, ignore_index=True)
    
        c = pd.concat([rc,sc], axis=0, keys=[False,True], names=['shuffled']).reset_index('shuffled')
        C = C.append(c, ignore_index=True)
    
    M.to_pickle(cacheFolder / 'stsw_m.pkl')
    P.to_pickle(cacheFolder / 'stsw_p.pkl')
    C.to_pickle(cacheFolder / 'stsw_c.pkl')
    

#%%
wAvg = (M.groupby(['shuffled','action','genotype','true','predicted'])
         .apply(analysisStaySwitchDecoding.wAvg, 'percent', 'noNeurons'))
wSem = (M.groupby(['shuffled','action','genotype','true','predicted'])
         .apply(analysisStaySwitchDecoding.bootstrap, 'percent', 'noNeurons'))

cm_df = pd.concat([wAvg, wSem], axis=1, keys=['mean','sem']).loc[False]

#%%
for (g,a), df in cm_df.groupby(['genotype','action']):
    ax = layout.axes['cm_{}_{}'.format(g,a)]['axis']
    cmap = sns.light_palette(style.getColor(a), as_cmap=True, reverse=False)
    
    cm_mean = df.unstack('predicted')['mean']
    cm_sem = df.unstack('predicted')['sem']
    cm_annot = np.apply_along_axis(lambda p: '{:.0%}\n±{:.0%}'.format(*p),
                                   2, np.stack([cm_mean, cm_sem],
                                               axis=2))
    
    sns.heatmap(cm_mean, annot=cm_annot, fmt='', cmap=cmap, square=True,
                xticklabels='', yticklabels='', cbar=False, vmin=0, vmax=1, 
                annot_kws={'fontsize':6, 'ha':'center', 'va':'center'}, ax=ax)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.axis('off')

cachedDataPath = cacheFolder / "staySwitchAcrossDays.pkl"
if cachedDataPath.is_file():
    decodingAcrossDays = pd.read_pickle(cachedDataPath)
else:
    decodingAcrossDays = analysisStaySwitchDecoding.decodeStaySwitchAcrossDays(endoDataPath, alignmentDataPath)
    decodingAcrossDays.to_pickle(cachedDataPath)

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecoding.svg")


#%%
#acc_df = P.loc[~P.label.str.endswith('o.')].copy() # only use win-stay, lose-switch trials
##incl_actions = ['mC2R','mL2C','mC2L','mR2C'] # which action to plot duration vs accuracy plots for?
#incl_actions = ['mC2R','mL2C','mC2L','mR2C']
#acc_df = acc_df.query('action in @incl_actions')
#
#acc_df['type'] = acc_df.label.str.slice(-2) # win-stay (r.) or lose-switch (o!)
#acc_groupby = acc_df.groupby(['type','shuffled','genotype','duration',
#                              'animal','date','noNeurons'])
#accuracy = acc_groupby.apply(lambda s: np.mean(s.prediction == s.label))
#confidence = acc_groupby.apply(lambda s: np.mean(s['r.'] / (s['r.'] + s['o!'])))
#confidence.loc['o!'] = 1 - confidence.loc[['o!']]
#observations = acc_groupby.size()
#acc_df = pd.concat([accuracy, observations], keys=['accuracy','observations'],
#                   axis=1).reset_index('noNeurons')
#%%



#%%
### compute summary data
##wAvg = (acc_df.groupby(['type','shuffled','genotype','duration'])
##              .apply(analysisStaySwitchDecoding.wAvg, 'accuracy', 'noNeurons'))
##wSem = (acc_df.groupby(['type','shuffled','genotype','duration'])
##              .apply(analysisStaySwitchDecoding.bootstrap, 'accuracy', 'noNeurons'))
##
##acc_sum_df = pd.concat([wAvg, wSem], axis=1, keys=['mean','sem'])
#
##%%
##data = (acc_df.query('duration >= .29 & duration <= .51')
##              .set_index(['noNeurons','observations'], append=True)
##              .unstack('shuffled')
##              .reset_index(['noNeurons','observations']))
##data['acc_diff'] = data['accuracy',False] - data['accuracy',True]
##
###%%
##fig,axs = plt.subplots(1,3, figsize=(10,5))
##for (g,d), df in data[['acc_diff','noNeurons']].groupby(['genotype','duration']):
##    ax = axs[{'d1':0,'a2a':1,'oprm1':2}[g]]
##
##    adata = df.dropna().unstack('type').dropna()
##    for r in adata.values:
##        ax.plot([d-.015, d+.015], [r[0],r[1]], lw=r[2]/400,
##                 c=style.getColor(g), alpha=.5)
##            
##    wAvg = df.groupby('type').apply(analysisStaySwitchDecoding.wAvg, 'acc_diff', 'noNeurons')
##    wSem = df.groupby('type').apply(analysisStaySwitchDecoding.bootstrap, 'acc_diff', 'noNeurons')
##    
##    ax.errorbar([d-.015,d+.015], wAvg.values, 3*wSem.values, lw=1.5,
##                color=style.getColor(g))
##    
##    ax.set_ylim((-1,1))
##        
##plt.show()
##
##%%
#data = (acc_df.query('duration >= .34 & duration <= .51')
#              .set_index('noNeurons', append=True))
#
#fig,axs = plt.subplots(1,3, figsize=(10,5))
#for (g,d,s), df in data.groupby(['genotype','duration','shuffled']):
#    ax = axs[{'d1':0,'a2a':1,'oprm1':2}[g]]
#    if not s:
#        adata = df.unstack('type').dropna().reset_index('noNeurons')
#        for r in adata.values:
#            ax.plot([d-.015, d+.015], [r[1],r[2]], lw=r[0]/200,
#                     c=style.getColor(g), alpha=.35)
#
##    if s:
##        adata = df.unstack('type').dropna()
##        for r in adata.values:
##            ax.plot([d-.015, d+.015], [r[0],r[1]], lw=r[2]/200,
##                     c=style.getColor('shuffled'), alpha=.25)
#    
#    df = df.reset_index('noNeurons')        
#    wAvg = df.groupby('type').apply(analysisStaySwitchDecoding.wAvg, 'accuracy', 'noNeurons')
#    wSem = df.groupby('type').apply(analysisStaySwitchDecoding.bootstrap, 'accuracy', 'noNeurons')
#    
#    ax.errorbar([d-.015,d+.015], wAvg.values, wSem.values, lw=1.5,
#                color=style.getColor(g) if not s else style.getColor('shuffled'))
#    
#    ax.axhline(.5, lw=mpl.rcParams['axes.linewidth'], ls=':', color='k', alpha=.5)
#    ax.set_ylim((0,1))
#        
#plt.show()
#
##%%
#axs = {}
#for gt in ['d1','a2a','oprm1']:
#    ax = layout.axes['acc_{}'.format(gt)]['axis']
#    axs[gt] = ax
#    data = adata.loc[gt]
#    summary = sdata.loc[gt]
#    
#    ax.scatter(analysisStaySwitchDecoding.jitter(data.duration, .0075), data[False,'accuracy'],
#               s=data[False,'noNeurons']/30, edgecolor=style.getColor(gt),
#               facecolor='none', alpha=.35, zorder=0, clip_on=False,
#               lw=mpl.rcParams['axes.linewidth'])
#    ax.errorbar(summary.duration, summary[False,'mean'], summary[False,'sem'],
#                color=style.getColor(gt))
#    ax.errorbar(summary.duration, summary[True,'mean'], summary[True,'sem'],
#                color=style.getColor('shuffled'))
#    ax.axhline(.5, ls=':', alpha=.5, color='k', lw=mpl.rcParams['axes.linewidth'])
#    
#    ax.set_ylim((0,1))
#    ax.yaxis.set_minor_locator(MultipleLocator(.25))
#    ax.set_yticks((0,.5,1))
#    ax.set_yticklabels(())
#    ax.set_xlim((.275,.525))
#    ax.xaxis.set_minor_locator(MultipleLocator(.05))
#    ax.set_xticks((.3,.4,.5))
#    sns.despine(ax=ax)
#
#axs['d1'].set_yticklabels((0,50,100))
#axs['d1'].set_ylabel('decoding accuracy (%)')
#axs['a2a'].set_xlabel('action duration (s)')
    
