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
#from matplotlib.ticker import MultipleLocator
import itertools

import analysisOpenField, analysisTunings
import style
from utils import readSessions, fancyViz #, alluvialPlot

style.set_context()
plt.ioff()

#%%

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
layout = figurefirst.FigureLayout(templateFolder / "followNeuronsSupp.svg")
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

labels = ["leftTurn", "mL2C", "mR2C", "all"] #"running", "mC2L", "mC2R"
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

for i in range(1,4):
    cax = layout.axes['colorbar_means_{}'.format(i)]['axis']
    cb = plt.colorbar(img, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cax.set_axis_off()
    cax.text(-0.325, -.1, "{:.1f}".format(-saturation), ha='right', va='center', fontdict={'fontsize':6})
    cax.text(0.325, -.1, "{:.1f}".format(saturation), ha='left', va='center', fontdict={'fontsize':6})
    cax.text(0, 0.5, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})


#%% Panel C
twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)
twoChoiceTunings = twoChoiceTunings.set_index(["animal", "date", "neuron"])[["action", "pct", "tuning"]]
joinedTunings = openFieldTunings.reset_index().join(twoChoiceTunings, on=["animal", "date", "neuron"], rsuffix="_2choice")
corrMeasure = lambda df: scipy.stats.pearsonr(df.tuning, df.tuning_2choice)[0]
correlations = joinedTunings.groupby(["genotype", "action", "action_2choice"]).apply(corrMeasure).unstack()

cax = layout.axes['corr_of2c_colorbar']['axis']
cax.tick_params(axis='y', which='both',length=0)

openFieldOrder = ["leftTurn", "rightTurn", "running", "stationary"]
order2choice = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "dL2C-", "pL2Co", "pL2Cr",
                "pC2L-", "pC2R-", "dR2C-", "pR2Co", "pR2Cr"]
for gt, perGt in correlations[order2choice].groupby(level=0):
    ax = layout.axes["openField2choiceCorrs_{}".format(gt)]["axis"]
    sns.heatmap(perGt.loc[gt].loc[openFieldOrder], ax=ax, vmin=-1, vmax=1, annot=True,
                fmt=".2f", cmap=cmocean.cm.balance, cbar=True, cbar_ax=cax,
                cbar_kws={'ticks':(-1,0,1)}, xticklabels=False, yticklabels=(gt=="d1"),
                annot_kws={'fontsize': 4.0},
                linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(genotypeNames[gt], pad=3)
    ax.set_ylim(4,0)
    ax.tick_params("both", length=0, pad=3)
layout.axes["openField2choiceCorrs_d1"]["axis"].set_yticklabels([behaviorNames[b] for b in openFieldOrder])

#%% Panel D
examples = [('d1','5643',('190112', '190114', '190128', '190130', '190201'), 170),
            ('a2a','6043',('190114', '190126', '190128', '190130', '190201'), 287),
            ('oprm1','5574',('190126', '190127', '190129', '190131', '190202'), 20)]
alignmentStore = h5py.File(alignmentDataPath, "r")
def findAlignedNeuron(genotype, animal, fromDate, toDate, neuron):
    if fromDate == toDate:
        return neuron
    else:
        matches = alignmentStore["/data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)]
        return pd.Series(matches[:,1], matches[:,0]).loc[neuron]

saturation = 1
for i in range(3):
    for j in range(5):
        sess = next(readSessions.findSessions(endoDataPath, animal=examples[i][1],
                                             date=examples[i][2][j], task="2choice"))
        neuron = findAlignedNeuron(examples[i][0], examples[i][1], examples[i][2][0],
                                   examples[i][2][j], examples[i][3])
        signal = sess.readDeconvolvedTraces()[neuron]
        signal -= signal.mean()
        signal /= signal.std()
        ax = layout.axes["acrossDays_ex{}{}".format(i+1,j+1)]["axis"]
        fv = fancyViz.SchematicIntensityPlot(sess, splitReturns=False,
                                             linewidth=mpl.rcParams['axes.linewidth'],
                                             saturation=saturation, smoothing=7)
        img = fv.draw(signal, ax=ax)
        
cax = layout.axes['colorbar_2choice_examples']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})
alignmentStore.close()

layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "followNeuronsSupp.svg")
