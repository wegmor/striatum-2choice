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
import scipy.spatial
import scipy.cluster
from sklearn.metrics import silhouette_samples
import style
plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "clusteringSupp.svg")
layout.make_mplfigures()

#%% Figure A
signalHistogram = analysisClusteringSupp.getSignalHistogram(endoDataPath)
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
varExplained = analysisClusteringSupp.getVarianceExplainedByPCA(endoDataPath)

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
topDim = analysisClusteringSupp.calculateTopologicalDimensionality(endoDataPath)
meanDim = topDim.groupby("session").mean()
color = [style.getColor(s.split("_")[0]) for s in meanDim.index]
ax = layout.axes['topological_dim']['axis']
ax.scatter(meanDim.nNeurons, meanDim.dimensionality, c=color)
ax.set_xlabel("number of neurons")
ax.set_ylabel("dimensionality")
ax.set_ylim(0, 100)
handles = [mpl.lines.Line2D([0], [0], ls='', marker='o', color=style.getColor(c)) for c in ("d1", "a2a", "oprm1")]
ax.legend(handles, ("D1", "A2A", "Oprm1"), loc="lower right")
sns.despine(ax=ax)

#%%
tuningData = analysisTunings.getTuningData(endoDataPath)
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
score_df = analysisClusteringSupp.getKMeansScores(tunings)

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
def toLongName(label):
    portNames = {'L': "left", 'R': "right", 'C': "center"}
    if label[0] == 'm':
        longName = "moving "
        longName += portNames[label[1]]
        longName += " to "
        longName += portNames[label[3]]
    elif label[:2] == "pC":
        longName = "center port going {}".format(portNames[label[3]])
    else:
        longName = portNames[label[1]]
        if label[4] == 'r':
            longName += " reward"
        elif label[4] == 'o':
            longName += " delay"
    longName += " ("
    longName = {'r': "win", 'o': "lose"}[label[4]]
    longName += "-"
    longName += {'.': "stay", '!': "switch"}[label[5]]
    #longName += ")"
    return longName

distMat = analysisClusteringSupp.populationDistMatrix(endoDataPath)
pdist = scipy.spatial.distance.squareform(distMat)
Z = scipy.cluster.hierarchy.linkage(pdist, 'ward')
labels = distMat.index
longLabels = list(map(toLongName, labels))
colors = []
for i in range(23):
    if Z[i, 3] <= 3: 
        color = style.getColor(labels[Z[i, :2].min()][:4])
        colors.append(mpl.colors.to_hex(color))
    else:
        colors.append("black")
plt.sca(layout.axes['dendrogram']['axis'])
dn = scipy.cluster.hierarchy.dendrogram(Z, labels=longLabels,
                                        link_color_func=lambda k: colors[k-24])
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
sns.despine(ax=plt.gca(), bottom=True, left=False, trim=False)
plt.ylim(0,80)
plt.ylabel("Distance")
plt.title("agglomerative clustering of pooled mean\npopulation activity in all task phases", pad=6)
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "clusteringSupp.svg")
