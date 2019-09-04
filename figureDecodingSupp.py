import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import h5py
import pathlib
import figurefirst
import cmocean

#import sys
#thisFolder = pathlib.Path(__file__).resolve().parent
#sys.path.append(str(thisFolder.parent))

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisDecoding
import style

style.set_context()
plt.ioff

#%%
dataFolder = pathlib.Path("data")
endoDataPath = dataFolder / "endoData_2019.hdf"
alignmentDataPath = dataFolder / "alignment_190227.hdf"

outputFolder = pathlib.Path("svg")
cacheFolder = pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "decodingSupp.svg")
layout.make_mplfigures()

#%% Panel A
cachedDataPath = cacheFolder / "decodeConfusion.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisDecoding.decodingConfusion(endoDataPath)
    decodingData.to_pickle(cachedDataPath)
order = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co", "pL2Cr",
         "pC2L-", "pC2R-", "pR2Cd", "pR2Co", "pR2Cr"]
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]

cax = layout.axes['dec_colorbar']['axis']
cax.tick_params(axis='both', which='both',length=0)

for gt, data in decodingData.groupby("genotype"):
    weightedData = data.set_index(["true", "predicted"]).eval("occurences * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    ax = layout.axes["confusionMatrix_{}".format(gt)]["axis"]
    sns.heatmap(weightedData[order].reindex(order), ax=ax, vmin=0, vmax=1, annot=True, fmt=".0%",
                cmap=cmocean.cm.amp, xticklabels=False, yticklabels=False,
                annot_kws={'fontsize': 4.5},  cbar=True, cbar_ax=cax,
                cbar_kws={'orientation':'horizontal', 'ticks':()},
                linewidths=matplotlib.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
cax.text(-.025, .25, 0, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, 100, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.1, 'accuracy (%)', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)


#%% Panel C
def calcCorr(df):
    r = scipy.stats.pearsonr(df.true, df.predicted)[0]
    return pd.Series((r, df.nNeurons.iloc[0]), ("correlation", "nNeurons"))

#titles = {'mC2L': 'Center-to-left', 'mC2R': 'Center-to-right',
#          'mL2C': 'Left-to-center', 'mR2C': 'Right-to-center'}
for label in ("mC2L", "mC2R", "mL2C", "mR2C","pC2L","pC2R"):
    cachedDataPath = cacheFolder / "decodeMovementProgress_{}.pkl".format(label)
    if cachedDataPath.is_file():
        decodingMovementProgress = pd.read_pickle(cachedDataPath)
    else:
        decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath, label=label+"-")
        decodingMovementProgress.to_pickle(cachedDataPath)
    avgCorr = decodingMovementProgress.groupby(["shuffle","sess"]).apply(calcCorr).reset_index("shuffle")
    avgCorr["genotype"] = avgCorr.index.str.split("_").str[0]
    avgCorr.loc[avgCorr.shuffle, "genotype"] = "shuffled"
    avgCorr["animal"] = avgCorr.index.str.split("_").str[1]
    avgCorr.loc[avgCorr.shuffle, "animal"] = avgCorr.loc[avgCorr.shuffle, "animal"] + "_r"
    avgCorr["date"] = avgCorr.index.str.split("_").str[2]
    avgCorr.sort_values(["genotype", "animal", "date"], ascending=False, inplace=True)
    
    ax = layout.axes["movementProgressCorrelations_{}".format(label)]["axis"]
    sessionBarPlot.sessionBarPlot(avgCorr, yCol="correlation", weightCol="nNeurons",
                                  genotypeOrder=('d1','a2a','oprm1','shuffled'),
                                  ax=ax, colorFunc=style.getColor, weightScale=0.035)
    
    ax.axhline(0, ls=':', alpha=.5, lw=matplotlib.rcParams['axes.linewidth'],
               color='k', clip_on=False)
    ax.set_xticks(())
    ax.set_yticks((0,.5,1))
    ax.set_yticks((.25,.75), minor=True)
    if label=="mL2C":
        ax.set_ylabel("true X predicted\ncorrelation")
    else:
        ax.set_yticklabels([])

    ax.set_ylim(0,1)
    sns.despine(ax=ax, bottom=True)
    
axt = layout.axes['bar_legend']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.3,
                                 label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1',
                                        'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']
                  ]
axt.legend(handles=legend_elements, ncol=4, loc='center',
           mode='expand')
axt.axis('off')


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "decodingSupp.svg")