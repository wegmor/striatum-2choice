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
from matplotlib.ticker import MultipleLocator

import analysisOpenField, analysisTunings
import style
from utils import readSessions, fancyViz, alluvialPlot

style.set_context()
plt.ioff()

#%%

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder = pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "openFieldSupp.svg")
layout.make_mplfigures()

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}

## Panel A
cachedDataPath = cacheFolder / "openFieldTunings.pkl"
if cachedDataPath.is_file():
    openFieldTunings = pd.read_pickle(cachedDataPath)
else:
    #All 4-behavior panels    
    cachedDataPath = cacheFolder / "segmentedBehavior.pkl"
    if cachedDataPath.is_file():
        segmentedBehavior = pd.read_pickle(cachedDataPath)
    else:
        segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
        segmentedBehavior.to_pickle(cachedDataPath)
    segmentedBehavior = segmentedBehavior.set_index("session")
    openFieldTunings = analysisOpenField.getTuningData(endoDataPath, segmentedBehavior)
    openFieldTunings.to_pickle(cachedDataPath)
    
cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    twoChoiceTunings = pd.read_pickle(cachedDataPath)
else:
    twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)
    twoChoiceTunings.to_pickle(cachedDataPath)
    
for t in (twoChoiceTunings, openFieldTunings):
    t['signp'] = t['pct'] > .995
    t['signn'] = t['pct'] < .005
    
primaryTwoChoice = twoChoiceTunings.loc[twoChoiceTunings.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
primaryTwoChoice.loc[~primaryTwoChoice.signp, 'action'] = 'none'
primaryOpenField = openFieldTunings.loc[openFieldTunings.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
primaryOpenField.loc[~primaryOpenField.signp, 'action'] = 'none'

primaryPairs = primaryOpenField.join(primaryTwoChoice.set_index(["animal", "date", "neuron"]),
                                     on=["animal", "date", "neuron"], rsuffix="_2choice", how="inner")
primaryPairs = primaryPairs[["genotype", "animal", "date", "neuron", "action", "action_2choice"]]
primaryPairs.rename(columns={'action': 'action_openField'}, inplace=True)

order_openField = ["stationary", "running", "leftTurn", "rightTurn", "none"]
order_twoChoice = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co", "pL2Cr",
                   "pC2L-", "pC2R-", "pR2Cd", "pR2Co", "pR2Cr", "none"]
primaryPairs.action_2choice = pd.Categorical(primaryPairs.action_2choice, order_twoChoice)
primaryPairs.action_openField = pd.Categorical(primaryPairs.action_openField, order_openField)

colormap = {a: style.getColor(a[:4]) for a in primaryPairs.action_2choice.unique()}
colormap.update({a: style.getColor(a) for a in primaryPairs.action_openField.unique()})

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}

for gt in ("d1", "a2a", "oprm1"):
    ax = layout.axes['alluvial_{}'.format(gt)]['axis']
    data = primaryPairs.query("genotype == '{}'".format(gt))
    alluvialPlot.alluvialPlot(data, "action_openField", "action_2choice",
                              colormap, ax, colorByRight=False, alpha=0.75)
    ax.set_xlim(-0.2,1.2)
    ax.axis("off")
    ax.set_title(genotypeNames[gt])

## Panel F
examples = [
    ('d1_5643_190201', 69),
    ('d1_5643_190201', 129),
    ('d1_5652_190203', 7),
    ('a2a_5693_190131', 121),
    ('a2a_5693_190202', 30),
    ('a2a_5693_190202', 132),
    ('oprm1_5308_190201', 97),
    ('oprm1_5308_190204', 182),
    ('oprm1_5703_190201', 16)
]
for i in range(9):
    genotype, animal, date = examples[i][0].split("_")
    twoChoiceSess = next(readSessions.findSessions(endoDataPath, animal=animal,
                                                   date=date, task="2choice"))
    openFieldSess = next(readSessions.findSessions(endoDataPath, animal=animal,
                                                   date=date, task="openField"))
    twoChoiceSignal = twoChoiceSess.readDeconvolvedTraces()[examples[i][1]]
    twoChoiceSignal -= twoChoiceSignal.mean()
    twoChoiceSignal /= twoChoiceSignal.std()
    openFieldSignal = openFieldSess.readDeconvolvedTraces()[examples[i][1]]
    openFieldSignal -= openFieldSignal.mean()
    openFieldSignal /= openFieldSignal.std()
    twoChoiceAx = layout.axes["ex_2choice_{}".format(i+1)]["axis"]
    openFieldAx = layout.axes["ex_of_{}".format(i+1)]["axis"]
    fv2choice = fancyViz.SchematicIntensityPlot(twoChoiceSess, linewidth=style.lw()*0.5,
                                                smoothing=7, splitReturns=False)
    img = fv2choice.draw(twoChoiceSignal, ax=twoChoiceAx)
    fvof = fancyViz.OpenFieldSchematicPlot(openFieldSess, linewidth=style.lw()*0.5)
    img = fvof.draw(openFieldSignal, ax=openFieldAx)
    
cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='vertical')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(.5, -.15, -1, ha='center', va='bottom', fontdict={'fontsize':6},
        transform=cax.transAxes)
cax.text(.5, 1.15, 1, ha='center', va='top', fontdict={'fontsize':6},
        transform=cax.transAxes)
cax.text(-1.05, 0.5, 'z-score', ha='center', va='center', fontdict={'fontsize':6},
        rotation=90, transform=cax.transAxes)

layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "openFieldSupp.svg")
