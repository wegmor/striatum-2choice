#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:02:59 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import analysisTunings, analysisClusteringSupport
import figurefirst
from sklearn.metrics import silhouette_samples
import style
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
cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisTunings.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)

tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005


#%%
layout = figurefirst.FigureLayout(templateFolder / "clusteringSupp.svg")
layout.make_mplfigures()


#%%
tunings = (tuningData.set_index(['genotype','animal','date','neuron','action'])['tuning']
                     .unstack('action'))
tunings = (tunings.loc[tuningData.groupby(['genotype','animal','date','neuron'])
                  .signp.sum() >= 1].copy())

#%% silhouette score for primary tuning clustering / genotype
silhouette_df = pd.DataFrame()
ax = layout.axes['tuning_silhouette']['axis']
for gt, gtTunings in tunings.groupby('genotype'):
    labels = gtTunings.idxmax(axis=1)
    silhouette = silhouette_samples(gtTunings, labels=labels)
    silhouette = pd.DataFrame({'coef':silhouette,
                               'label':labels},
                               index=gtTunings.index)
    silhouette_df = silhouette_df.append(silhouette.reset_index(),
                                         ignore_index=True)

    sns.distplot(silhouette.coef, bins=np.arange(-1,1.1,.1),
                 ax=ax, color=style.getColor(gt), hist=False,
                 kde_kws={'clip_on':False, 'alpha':.75}, label=gt)

ax.set_xlim((-1,1))
ax.set_ylim((0,4))
ax.set_yticks((0,2,4))
ax.set_yticks((1,3), minor=True)
ax.axvline(0, color='k', ls=':', zorder=-99, alpha=.5, lw=mpl.rcParams['axes.linewidth'])
ax.set_xticks(())
ax.set_xlabel('')
ax.set_ylabel('density')
ax.legend(['A2A','D1','Oprm1'], loc='upper left', bbox_to_anchor=(.06,.95))
sns.despine(bottom=True, trim=True, ax=ax)


ax = layout.axes['tuning_silhouette_bp']['axis']
palette = {gt: style.getColor(gt) for gt in ['d1','a2a','oprm1']}

sns.boxplot('coef', 'genotype', data=silhouette_df, ax=ax, 
            palette=palette, saturation=.85, showcaps=False, showfliers=False,
            boxprops={'alpha':0.75, 'lw':0, 'zorder':-99, 'clip_on':False}, 
            width=.75, whiskerprops={'c':'k','zorder':99, 'clip_on':False},
            medianprops={'c':'k','zorder':99, 'clip_on':False})

ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
ax.set_xlim((-1,1))
ax.set_ylim((-.75,2.75))
ax.set_xticks((-1,0,1))
ax.set_xticks((-.5,.5), minor=True)
ax.set_xlabel('silhouette coefficient')
ax.set_yticks(())
ax.set_ylabel('')
sns.despine(left=True, trim=True, ax=ax)


#%% silhouette score for genotype clustering for pooled neurons
ax = layout.axes['tuning_silhouette_gt']['axis']
labels = tunings.idxmax(axis=1)
silhouette = silhouette_samples(tunings, labels=labels)
silhouette = pd.DataFrame({'coef':silhouette,
                           'label':labels},
                           index=tunings.index)
silhouette = silhouette.reset_index()

sns.distplot(silhouette.coef, bins=np.arange(-1,1.1,.05),
             ax=ax, color='k', hist=False,
             kde_kws={'clip_on':False, 'alpha':1}, label='pooled')
sns.distplot(silhouette.coef, bins=np.arange(-1,1.1,.05),
             ax=ax, color='k', hist=True, kde=False,
             hist_kws={'alpha':.35}, norm_hist=True)

ax.set_xlim((-1,1))
ax.set_ylim((0,4))
ax.set_yticks((0,2,4))
ax.set_yticks((1,3), minor=True)
ax.axvline(0, color='k', ls=':', zorder=-99, alpha=.5, lw=mpl.rcParams['axes.linewidth'])
ax.set_xticks(())
ax.set_xlabel('')
ax.set_ylabel('density')
ax.legend(loc='upper left', bbox_to_anchor=(0.06,.95))
sns.despine(bottom=True, trim=True, ax=ax)


ax = layout.axes['tuning_silhouette_gt_bp']['axis']

sns.boxplot('coef', data=silhouette_df, ax=ax, color='k',
            saturation=.85, showcaps=False, showfliers=False,
            boxprops={'alpha':0.35, 'lw':0, 'zorder':-99, 'clip_on':False}, 
            width=.75, whiskerprops={'c':'k','zorder':99, 'clip_on':False},
            medianprops={'c':'k','zorder':99, 'clip_on':False})

ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
ax.set_xlim((-1,1))
ax.set_ylim((-1.2,1.2))
ax.set_xticks((-1,0,1))
ax.set_xticks((-.5,.5), minor=True)
ax.set_xlabel('silhouette coefficient')
ax.set_yticks(())
ax.set_ylabel('')
sns.despine(left=True, trim=True, ax=ax)


#%%
cachedDataPath = cacheFolder / "silhouette_score_df.pkl"
if cachedDataPath.is_file():
    score_df = pd.read_pickle(cachedDataPath)
else:
    score_df = analysisClusteringSupport.getKMeansScores(tunings)
    score_df.to_pickle(cachedDataPath)


#%%
ax = layout.axes['tuning_score']['axis']
for gt, gt_scores in score_df.groupby('genotype'):
    #ax = layout.axes['tuning_score_'+gt]['axis']
    ax.fill_between(gt_scores.n_clusters,
                    gt_scores.score_avg - gt_scores.score_std,
                    gt_scores.score_avg + gt_scores.score_std,
                    color = style.getColor(gt), alpha=.25,
                    lw=0)
    ax.plot(gt_scores.n_clusters, gt_scores.score_avg,
            color=style.getColor(gt), label=gt)
#    ax.fill_between(gt_scores.n_clusters,
#                    gt_scores.score_shuffle_avg - gt_scores.score_shuffle_std,
#                    gt_scores.score_shuffle_avg + gt_scores.score_shuffle_std,
#                    color = style.getColor('shuffled'), alpha=.35)
#    ax.plot(gt_scores.n_clusters, gt_scores.score_shuffle_avg,
#            color=style.getColor('shuffled'))
    ax.set_ylim((0,.3))
    ax.set_yticks((0,.1,.2,.3))
    ax.set_xlim((0,50))
    ax.set_xticks((2,25,50))
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('silhouette score')
    ax.legend(labels=['A2A','D1','Oprm1'], loc='upper right', bbox_to_anchor=(1,.95))
    sns.despine(ax=ax)
        

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "clusteringSupp.svg")