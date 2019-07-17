import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
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

endoDataPath = "endoData_2019.hdf"
alignmentDataPath = "alignment_190227.hdf"

outputFolder = pathlib.Path("svg")
cacheFolder = pathlib.Path("cache")
templateFolder = pathlib.Path(__file__).parent / "templates"

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "decodingSupp.svg")
layout.make_mplfigures()

## Panel A
cachedDataPath = cacheFolder / "decodeConfusion.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisDecoding.decodingConfusion(endoDataPath)
    decodingData.to_pickle(cachedDataPath)
order = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co", "pL2Cr",
         "pC2L-", "pC2R-", "pR2Cd", "pR2Co", "pR2Cr"]
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]
for gt, data in decodingData.groupby("genotype"):
    weightedData = data.set_index(["true", "predicted"]).eval("occurencies * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    ax = layout.axes["confusionMatrix_{}".format(gt)]["axis"]
    sns.heatmap(weightedData[order].reindex(order), ax=ax, vmin=0, vmax=1, annot=True, fmt=".0%", cmap=cmocean.cm.amp,
                cbar=False, xticklabels=False, yticklabels=False, annot_kws={'fontsize': 4.5},
                linewidths=matplotlib.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

## Panel B
def calcCorr(df):
    r = scipy.stats.pearsonr(df.true, df.predicted)[0]
    return pd.Series((r, df.nNeurons.iloc[0]), ("correlation", "nNeurons"))
titles = {'mC2L': 'Center-to-left', 'mC2R': 'Center-to-right',
          'mL2C': 'Left-to-center', 'mR2C': 'Right-to-center'}
for label in ("mC2L", "mC2R", "mL2C", "mR2C"):
    cachedDataPath = cacheFolder / "decodeMovementProgress_{}.pkl".format(label)
    if cachedDataPath.is_file():
        decodingMovementProgress = pd.read_pickle(cachedDataPath)
    else:
        decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath, label=label+"-")
        decodingMovementProgress.to_pickle(cachedDataPath)
    avgCorr = decodingMovementProgress.query("not shuffle").groupby("sess").apply(calcCorr)
    avgCorr["genotype"] = avgCorr.index.str.split("_").str[0]
    avgCorr["animal"] = avgCorr.index.str.split("_").str[1]
    avgCorr["date"] = avgCorr.index.str.split("_").str[2]
    avgCorr.sort_values(["genotype", "animal", "date"], ascending=False, inplace=True)
    ax = layout.axes["movementProgressCorrelations_{}".format(label)]["axis"]
    sessionBarPlot.sessionBarPlot(avgCorr, yCol="correlation", weightCol="nNeurons",
                                  ax=ax, colorFunc=style.getColor, weightScale=0.05)
    if label=="mC2L":
        ax.set_ylabel("Correlation\ntruth and decoded")
    else:
        ax.set_yticklabels([])
    ax.set_title(titles[label])
    ax.set_ylim(0,1)
    sns.despine(ax=ax)
    
layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "decodingSupp.svg")