import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import analysisTunings
import figurefirst
import itertools
import tqdm
from utils import readSessions, fancyViz
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
layout = figurefirst.FigureLayout(templateFolder / "averageActivityPerTuning.svg")
layout.make_mplfigures()

#%% Figures A and B
cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisTunings.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)
    
tunings = tuningData.set_index(["animal", "date", "neuron", "action"]).pct > 0.995
tunings = tunings.unstack().sort_index()


labels = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co",
          "pL2Cr", "pC2L-", "pC2R-", "pR2Cd", "pR2Co","pR2Cr"]
saturation = 0.4
fvs = {}
for key in itertools.product(('d1', 'a2a', 'oprm1'), labels, (False, True)):
    fvs[key] = fancyViz.SchematicIntensityPlot(splitReturns=False, 
                                               linewidth=mpl.rcParams['axes.linewidth'],
                                               smoothing=7, saturation=saturation)
for sess in tqdm.tqdm(readSessions.findSessions(endoDataPath, task="2choice"), total=66):
    lfa = sess.labelFrameActions(reward="sidePorts", switch=False)
    traces = sess.readDeconvolvedTraces(zScore=True)
    if len(traces) != len(lfa): continue
    genotype = sess.meta.genotype
    for label in labels:
        fvs[(genotype, label, False)].setSession(sess)
        fvs[(genotype, label, True)].setSession(sess)
        tuned = tunings.loc[(sess.meta.animal, sess.meta.date)][label]
        single = tunings.loc[(sess.meta.animal, sess.meta.date)].sum(axis=1)==1
        for neuron in traces.columns:
            if tuned[neuron]:
                fvs[(genotype, label, False)].addTraceToBuffer(traces[neuron])
                if single[neuron]:
                    fvs[(genotype, label, True)].addTraceToBuffer(traces[neuron])

for gt, l in itertools.product(("d1", "a2a", "oprm1"), labels):
    if l[-1] == '-': axName = l[:4]+"_"+gt
    else: axName = l+"_"+gt
    
    #Figure A
    fvs[(gt, l, False)].drawBuffer(ax=layout.axes[axName]['axis'])
    
    #Figure B
    fvs[(gt, l, True)].drawBuffer(ax=layout.axes[axName+"_single"]['axis'])

for single in (False, True):
    cax = layout.axes['colorbar'+('_single' if single else '')]['axis']
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cax.set_axis_off()
    cax.text(-0.05, 0.5, str(-saturation), ha='right', va='center',
             fontdict={'fontsize':6}, transform=cax.transAxes)
    cax.text(1.05, 0.5, str(saturation), ha='left', va='center',
             fontdict={'fontsize':6}, transform=cax.transAxes)
    cax.text(0.5, 1.1, 'z-score', ha='center', va='bottom',
             fontdict={'fontsize':6}, transform=cax.transAxes)

layout.insert_figures('plots')
layout.write_svg(outputFolder / "averageActivityPerTuning.svg")
