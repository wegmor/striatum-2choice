#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:59:09 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import figurefirst
#import cmocean
from scipy.stats import pearsonr
from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisDecoding
import analysisStaySwitchDecoding
import style
from matplotlib import lines as mlines

style.set_context()
plt.ioff()


#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path('svg')
cacheFolder = pathlib.Path('cache')
templateFolder = pathlib.Path('templates')

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()
    
    
#%%
layout = figurefirst.FigureLayout(templateFolder / "turnDecoding.svg")
layout.make_mplfigures()


#%% Panel A
exampleNeurons = (7, 66, 13)
saturation = 1

sess = next(readSessions.findSessions(endoDataPath, animal="5308", date="190131"))
lfa = sess.labelFrameActions(reward="sidePorts")
deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
X = deconv[lfa.label=="mR2C-"]
Y = lfa.actionProgress[lfa.label=="mR2C-"]
avgActivity = X.groupby((Y*10).astype("int")/10.0).mean().T
sorting = avgActivity.idxmax(axis=1).argsort()
plt.sca(layout.axes["movementProgressRaster"]["axis"])
plt.imshow(avgActivity.iloc[sorting], aspect="auto",
           interpolation="nearest", vmin=-saturation, vmax=saturation, cmap="RdYlBu_r")
markY = [dict(sorting.reset_index().values[:,::-1])[n] for n in exampleNeurons]
plt.xlim((-.5,9.5))
plt.scatter([-.8]*3, markY, marker='.', s=.25, c='k', clip_on=False,
            linewidths=0)
plt.xticks((0,9), ('R', 'C'))#, rotation=30, ha="right", va="top")#(0, 50, 100))
plt.yticks([0, len(sorting)-1], [len(sorting), 1])
plt.xlabel("turn progress")
plt.ylabel('neuron', labelpad=-10)
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position('right')
plt.gca().tick_params(axis='both', which='both',length=0)
sns.despine(ax=plt.gca(), top=True, bottom=True, left=True, right=True)

fv = fancyViz.SchematicIntensityPlot(sess, linewidth=mpl.rcParams['axes.linewidth'],
                                     splitReturns=False, smoothing=7, saturation=saturation)
for i in range(3):
    ax = layout.axes["movementExample{}".format(i+1)]["axis"]
    img = fv.draw(deconv[exampleNeurons[i]], ax=ax)
    
cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-.025, .25, -saturation, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, saturation, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)

    
#%% Panel B
cachedDataPath = cacheFolder / "decodeMovementProgress_mR2C.pkl"
if cachedDataPath.is_file():
    decodingMovementProgress = pd.read_pickle(cachedDataPath)
else:
    decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath)
    decodingMovementProgress.to_pickle(cachedDataPath)
    
decodingMovementProgress['genotype'] = decodingMovementProgress.sess.str.split('_').str[0]
decodingMovementProgress['animal'] = decodingMovementProgress.sess.str.split('_').str[1]
decodingMovementProgress['date'] = decodingMovementProgress.sess.str.split('_').str[2]

    
def calcCorr(df):
    r = pearsonr(df.true, df.predicted)[0]
    return pd.Series((r, df.nNeurons.iloc[0]), ("correlation", "nNeurons"))

exampleSession = decodingMovementProgress.query("sess == 'oprm1_5308_190131' & not shuffle")
means = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.mean()
stds = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.std()
plt.sca(layout.axes["decodingProgressExample"]["axis"])
plt.plot([0,100], [0, 100], color='k', ls=':', alpha=0.5, lw=mpl.rcParams['axes.linewidth'])
plt.errorbar(means.index*100, means*100, yerr=stds*100, fmt='o-', ms=3.2,
             color=style.getColor("oprm1"), markeredgewidth=0)
plt.xlim(-5,100)
plt.ylim(-5,100)
plt.xticks((0,50,100))#, rotation=30, ha="right", va="top")
plt.yticks((0,50,100))
plt.xlabel("true")
plt.ylabel("predicted", labelpad=-2.25)
corr = calcCorr(exampleSession).loc["correlation"]
plt.text(100, 1, "r = {:.3f}".format(corr), fontsize=mpl.rcParams['font.size'],
         color="k", ha="right", va='center')
sns.despine(ax=plt.gca())


#%% Panel C
avgCorr = decodingMovementProgress.groupby(['shuffle','genotype','animal','date']).apply(calcCorr)
avgCorr = avgCorr.unstack('shuffle')
avgCorr.columns = avgCorr.columns.reorder_levels((1,0))

for gt, gdata in avgCorr.groupby('genotype'):
    ax = layout.axes['{}_move_corr'.format(gt)]['axis']
    
    wAvg = analysisStaySwitchDecoding.wAvg(gdata[False], 'correlation', 'nNeurons')
    wSem = analysisStaySwitchDecoding.bootstrap(gdata[False], 'correlation', 'nNeurons')
    r_wAvg = analysisStaySwitchDecoding.wAvg(gdata[True], 'correlation', 'nNeurons')
    r_wSem = analysisStaySwitchDecoding.bootstrap(gdata[True], 'correlation', 'nNeurons')
    
    ax.errorbar(0, wAvg, yerr=wSem, color=style.getColor(gt), clip_on=False,
                marker='v', markersize=3.6, markerfacecolor='w',
                markeredgewidth=.8)
    ax.errorbar(1, r_wAvg, yerr=r_wSem, color=style.getColor(gt), clip_on=False,
                marker='o', markersize=3.2, markerfacecolor='w',
                markeredgewidth=.8)
    ax.plot([0,1], [wAvg, r_wAvg], color=style.getColor(gt), clip_on=False)
    
    for corr in gdata.values:
        ax.plot([0,1], corr[:2], lw=mpl.rcParams['axes.linewidth'], alpha=.2,
                clip_on=False, zorder=-99, color=style.getColor(gt))
    
    ax.axhline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])

    ax.set_ylim((0,.6))
    ax.set_xlim((-.35,1.35))
    ax.set_xticks(())
    ax.set_yticks((0,.6))
    ax.set_yticks((.3,), minor=True)
    if gt == 'a2a':
        ax.set_ylabel('true X predicted correlation')
    sns.despine(ax=ax, bottom=True, trim=True)
    
ax = layout.axes['corr_legend']['axis']
legend_elements = [mlines.Line2D([0], [0], marker='o', color='k',
                                 label='shuffled\ndecoder',
                                 markerfacecolor='w', markersize=3.2,
                                 markeredgewidth=.8)
                  ]
ax.legend(handles=legend_elements, loc='center')
ax.axis('off')


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "turnDecoding.svg")

