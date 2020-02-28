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
import itertools

import analysisOpenField, analysisTunings
import style
from utils import readSessions, fancyViz, alluvialPlot

style.set_context()
plt.ioff()

#%%

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
layout = figurefirst.FigureLayout(templateFolder / "openFieldNewSupp.svg")
layout.make_mplfigures()

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}

#%% Panel A
examples = {
    'd1':   [('d1_5643_190201', 69), ('d1_5643_190201', 129), ('d1_5652_190203', 7)],
    'a2a':  [('a2a_5693_190131', 121), ('a2a_5693_190202', 30), ('a2a_5693_190202', 132)],
    'oprm1':[('oprm1_5308_190201', 97), ('oprm1_5308_190204', 182), ('oprm1_5703_190201', 16)]
}
for gt in ("d1", "a2a", "oprm1"):
    for i in range(3):
        genotype, animal, date = examples[gt][i][0].split("_")
        neuron = examples[gt][i][1]
        twoChoiceSess = next(readSessions.findSessions(endoDataPath, animal=animal,
                                                       date=date, task="2choice"))
        openFieldSess = next(readSessions.findSessions(endoDataPath, animal=animal,
                                                       date=date, task="openField"))
        twoChoiceSignal = twoChoiceSess.readDeconvolvedTraces()[neuron]
        twoChoiceSignal -= twoChoiceSignal.mean()
        twoChoiceSignal /= twoChoiceSignal.std()
        openFieldSignal = openFieldSess.readDeconvolvedTraces()[neuron]
        openFieldSignal -= openFieldSignal.mean()
        openFieldSignal /= openFieldSignal.std()
        twoChoiceAx = layout.axes["ex_2choice_{}_{}".format(gt, i+1)]["axis"]
        openFieldAx = layout.axes["ex_of_{}_{}".format(gt, i+1)]["axis"]
        fv2choice = fancyViz.SchematicIntensityPlot(twoChoiceSess, linewidth=style.lw()*0.5,
                                                    smoothing=7, splitReturns=False)
        img = fv2choice.draw(twoChoiceSignal, ax=twoChoiceAx)
        fvof = fancyViz.OpenFieldSchematicPlot(openFieldSess, linewidth=style.lw()*0.5)
        img = fvof.draw(openFieldSignal, ax=openFieldAx)
    
cax = layout.axes['colorbar_examples']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})

#%% Panel B
openFieldTunings = analysisOpenField.getTuningData(endoDataPath)
twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)
for t in (twoChoiceTunings, openFieldTunings):
    t['signp'] = t['pct'] > .995
    t.set_index(["animal", "date", "action", "neuron"], inplace=True)
    t.sort_index(inplace=True)

labels = ["leftTurn", "running", "mC2L", "mC2R", "mL2C", "mR2C", "all"]
saturation = 0.3
fvs = {}
for key in itertools.product(('d1', 'a2a', 'oprm1'), labels):
    fvs[("2choice",)+key] = fancyViz.SchematicIntensityPlot(splitReturns=False,
                                linewidth=mpl.rcParams['axes.linewidth'],
                                smoothing=5, saturation=saturation)
    
for key in itertools.product(('d1', 'a2a', 'oprm1'), labels):
    fvs[("openField",)+key] = fancyViz.OpenFieldSchematicPlot(
                                linewidth=mpl.rcParams['axes.linewidth'],
                                smoothing=3, saturation=saturation)

for sess in readSessions.findSessions(endoDataPath, task=["2choice", "openField"]):
    shortKey = (sess.meta.animal, sess.meta.date)
    if shortKey not in openFieldTunings.index: continue
    if shortKey not in twoChoiceTunings.index: continue
    traces = sess.readDeconvolvedTraces(zScore=True)
    genotype = sess.meta.genotype
    task = sess.meta.task
    for label in labels:
        fvs[(task, genotype, label)].setSession(sess)
        if label == "all": #Don't look at tuning
            tuned = np.full(traces.shape[1], True)
        elif label[0] == 'm': #2-choice tuning
            tuned = twoChoiceTunings.loc[shortKey+(label+"-",)].signp
        else: #Open field tuning
            tuned = openFieldTunings.loc[shortKey+(label,)].signp
        for neuron in traces.columns:
            if tuned[neuron]:
                fvs[(task, genotype, label)].addTraceToBuffer(traces[neuron])
for task, gt, label in itertools.product(("openField", "2choice"),
                                         ("d1", "a2a", "oprm1"), labels):
    axName = "_".join(("mean", label, "of" if task=="openField" else task, gt))
    img = fvs[(task, gt, label)].drawBuffer(ax=layout.axes[axName]['axis'])

cax = layout.axes['colorbar_means']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-0.325, -.1, "{:.1f}".format(-saturation), ha='right', va='center', fontdict={'fontsize':6})
cax.text(0.325, -.1, "{:.1f}".format(saturation), ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 0.5, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})

layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "openFieldNewSupp.svg")
