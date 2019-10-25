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
import analysisTunings, analysisClusteringSupp
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
layout = figurefirst.FigureLayout(templateFolder / "clusteringSupp.svg")
layout.make_mplfigures()

#%% Figure A
cachedDataPath = cacheFolder / "signalHistogram.pkl"
if cachedDataPath.is_file():
    signalHistogram = pd.read_pickle(cachedDataPath)
else:
    signalHistogram = analysisClusteringSupp.getSignalHistogram(endoDataPath)
    signalHistogram.to_pickle(cachedDataPath)
ax = layout.axes['value_distribution']['axis']
ax.set_xscale("log")
norm = signalHistogram.iloc[1:].sum() * 0.05
ax.fill_between(signalHistogram.iloc[1:].index, 0, signalHistogram.iloc[1:] / norm, color="C1")
ax.set_xlabel("signal value (sd)")
ax.set_ylabel("pdf")
ax.set_xlim(10**-2, 10**1.5)
ax.set_ylim(0, 1.2)
sns.despine(ax=ax)
ax = layout.axes['inset_pie']['axis']
ax.pie([signalHistogram.iloc[0], signalHistogram.iloc[1:].sum()], labels=["$\leq0$", "$>0$"])

#%% Figure B
cachedDataPath = cacheFolder / "varExplainedByPCA.pkl"
if cachedDataPath.is_file():
    varExplained = pd.read_pickle(cachedDataPath)
else:
    varExplained = analysisClusteringSupp.getVarianceExplainedByPCA(endoDataPath)
    varExplained.to_pickle(cachedDataPath)
    
ax = layout.axes['naive_pca']['axis']

unsmoothed = varExplained.query("sigma == 0")
for s, d in unsmoothed.groupby("session"):
    col = style.getColor(s.split("_")[0])
    fracIncluded = d.numNeuronsIncluded / d.totNeurons
    ax.plot(100*fracIncluded, 100*d.fracVarExplained, c=col, alpha=0.1, lw=0.5)
binnedFrac = pd.cut(unsmoothed.numNeuronsIncluded / unsmoothed.totNeurons, np.linspace(0,1,21),
                    include_lowest=True, labels=False)
genotypes = unsmoothed.session.str.split("_").str[0]
for gt, means in unsmoothed.groupby([genotypes, binnedFrac]).fracVarExplained.mean().groupby(level=0):
    ax.plot(np.arange(2.5,100, 5), means*100, c=style.getColor(gt), lw=1)
ax.set_xlabel("PCs included (%)")
ax.set_ylabel("variance explained (%)")
handles = [mpl.lines.Line2D([0], [0], lw=1, c=style.getColor(c)) for c in ("d1", "a2a", "oprm1")]
ax.legend(handles, ("D1", "A2A", "Oprm1"), loc="lower right")
ax.set_xlim(0,100)
ax.set_ylim(0,100)
sns.despine(ax=ax)

#%% Figure C
neuronsTo90 = unsmoothed.query("fracVarExplained >= 0.9").groupby("session").numNeuronsIncluded.min()
totNeurons = unsmoothed.groupby("session").totNeurons.first()
genotypes = unsmoothed.groupby("session").session.first().str.split("_").str[0]
ax = layout.axes['pcs_required']['axis']
ax.scatter(totNeurons, neuronsTo90, c=list(map(style.getColor, genotypes)))
ax.set_xlabel("neurons in session")
ax.set_ylabel("number of PCs")
ax.set_title("PCs required to explain\n90% of the variance")
ax.set_xlim(0, 700)
ax.set_ylim(0, 700)
handles = [mpl.lines.Line2D([0], [0], ls='', marker='o', color=style.getColor(c)) for c in ("d1", "a2a", "oprm1")]
ax.legend(handles, ("D1", "A2A", "Oprm1"), loc="lower right")
sns.despine(ax=ax)

#%% Figure D
ax = layout.axes['smoothed_pca']['axis']
binnedFrac = pd.cut(varExplained.numNeuronsIncluded / varExplained.totNeurons,
                    np.linspace(0,1,21), include_lowest=True, labels=False)
for sigma, means in varExplained.groupby(["sigma", binnedFrac]).fracVarExplained.mean().groupby(level=0):
    ax.plot(np.arange(2.5,100, 5), means*100, label="$\sigma={}$ms".format(sigma*50))
ax.legend(title="smoothing")
ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_xlabel("PCs included (%)")
ax.set_ylabel("variance explained (%)")
sns.despine(ax=ax)

#%% Figure E
cachedDataPath = cacheFolder / "topologicalDimensionality.pkl"
if cachedDataPath.is_file():
    topDim = pd.read_pickle(cachedDataPath)
else:
    topDim = analysisClusteringSupp.calculateTopologicalDimensionality(endoDataPath)
    topDim.to_pickle(cachedDataPath)
    
meanDim = topDim.groupby("session").mean()
color = [style.getColor(s.split("_")[0]) for s in meanDim.index]
ax = layout.axes['topological_dim']['axis']
ax.scatter(meanDim.nNeurons, meanDim.dimensionality, c=color)
ax.set_xlabel("number of neurons")
ax.set_ylabel("dimensionality")
ax.set_ylim(0, 100)
sns.despine(ax=ax)

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
    score_df = analysisClusteringSupp.getKMeansScores(tunings)
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
