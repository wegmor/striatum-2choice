import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pathlib
import figurefirst
import cmocean
import tqdm

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

layout = figurefirst.FigureLayout(templateFolder / "tuningsSupp.svg")
layout.make_mplfigures()

## Panel B
cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisTunings.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)

order = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co", "pL2Cr",
         "pC2L-", "pC2R-", "pR2Cd", "pR2Co", "pR2Cr"]
tunings = tuningData.set_index(["genotype", "animal", "date", "neuron", "action"]).tuning
for genotype in ("oprm1", "d1", "a2a"):
    corr = tunings.loc[genotype].unstack()[order].corr()
    ax = layout.axes["correlationMatrix_{}".format(genotype)]["axis"]
    sns.heatmap(corr, ax=ax, vmin=0, vmax=1, annot=True, fmt=".2f", cmap=cmocean.cm.balance,
                cbar=False, xticklabels=False, yticklabels=False, annot_kws={'fontsize': 4.0},
                linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

## Panel C
meanPlots = {g: fancyViz.SchematicIntensityPlot(splitReturns=False,
                                                linewidth=mpl.rcParams['axes.linewidth'],
                                                smoothing=7) for g in ("oprm1", "d1", "a2a")}
for sess in readSessions.findSessions("endoData_2019.hdf", task="2choice"):
    signal = sess.readDeconvolvedTraces(zScore=True)
    if len(signal) != len(sess.readSensorValues()):
        continue
    genotype = sess.meta.genotype
    meanPlots[genotype].setSession(sess)
    for neuron in tqdm.tqdm(signal.columns, desc=str(sess)):
        meanPlots[genotype].addTraceToBuffer(signal[neuron])
        
for genotype, meanPlot in meanPlots.items():
    ax = layout.axes["genotypeAvg_{}".format(genotype)]["axis"]
    meanPlot.drawBuffer(ax=ax)
    
layout.insert_figures('plots')
layout.write_svg(outputFolder / "tuningsSupp.svg")