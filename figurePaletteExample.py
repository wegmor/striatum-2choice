#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:15:03 2019

@author: mowe
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FixedLocator
import pathlib
import figurefirst
import style
import analysisStaySwitchDecoding, analysisTunings, analysisOpenField
import cmocean
from collections import defaultdict
#from scipy.stats import ttest_1samp
import scipy.stats
#import statsmodels.formula.api as smf
from scipy.spatial.distance import pdist, squareform
from utils import readSessions, fancyViz, sessionBarPlot, alluvialPlot, roiPlot
plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path("data") / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder = pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()


#%%
layout = figurefirst.FigureLayout(templateFolder / "paletteExampleFigure.svg")
layout.make_mplfigures()

#%%
ax = layout.axes['bigLegend']['axis']
labels = [("d1", 1.0, "D1"), ("a2a", 1.0, "A2A"), ("oprm1", 1.0, "Oprm1"),
          ("shuffled", 1.0, "shuffled"),
          ("leftTurn", 0.15, "left turn"), ("rightTurn", 0.15, "right turn"),
          ("running", 0.15, "running"), ("stationary", 0.0, "stationary"),
          ("leftTurn", 1.0, "left turn tuning"), ("rightTurn", 1.0, "right turn tuning"),
          ("running", 1.0, "running tuning"), ("stationary", 1.0, "stationary tuning"),
          ("pL", 0.15, "in left port"), ("pR", 0.15, "in right port"),
          ("pC", 0.15, "in center port"), ("blank", 0.15, "between ports"),
          ("mC2L", 1.0, "center-to-left tuning"), ("mC2R", 1.0, "center-to-right tuning"),
          ("mL2C", 1.0, "left-to-center tuning"), ("mR2C", 1.0, "right-to-center tuning"),
          ("pC2L", 0.45, "center port tuning (left)"), ("pC2R", 1.0, "center port tuning (right)"),
          ("pL2C", 1.0, "left port tuning"), ("pR2C", 1.0, "right port tuning"),
          ("r.", 1.0, "win-stay"), ("o.", 1.0, "lose-stay"), ("o!", 1.0, "lose-switch")]
legend_elements = [mpatches.Patch(color=style.getColor(k), alpha=a,
                                 label=l) for k,a,l in labels]
ax.legend(handles=legend_elements, ncol=7, loc='center',
           mode='expand')
ax.axis("off")

#%%
longGtNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running',
                 'leftTurn': 'left turn', 'rightTurn': 'right turn'}
behaviorOrder = ("leftTurn", "rightTurn", "running", "stationary")
genotypeOrder = ("d1", "a2a", "oprm1")
tuningData = analysisOpenField.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005
exampleSessParams = {'genotype':'oprm1', 'animal':'5308', 'date':'190224'}
exampleSess = next(readSessions.findSessions(endoDataPath, task='openField', **exampleSessParams))
traces = exampleSess.readDeconvolvedTraces(rScore=True)
queryStr = " & ".join(["{}=='{}'".format(*v) for v in exampleSessParams.items()])
ex_tunings = tuningData.query(queryStr)

#%% Example path
coords = analysisOpenField.getSmoothedOFTTracking(endoDataPath, **exampleSessParams)
ax = layout.axes['ex_path']['axis']
start, end = 3200, 3200+2400
cdf = coords.loc[start:end].copy()
ax.plot(cdf.x, cdf.y, color='k', lw=0.5, zorder=0)
for (actioNo, behavior), adf in cdf.groupby(['actionNo', 'behavior']):
    ax.plot(adf.x, adf.y, c=style.getColor(behavior), lw=5, alpha=.43,
                 zorder=2)
ax.set_xticks(())
ax.set_yticks(())
ax.set_xlim(0,49)
ax.set_ylim(0,49)

#%% Example traces and fancyViz
best_neurons = ex_tunings.set_index("neuron").groupby("action").tuning.idxmax()
order = ["leftTurn", "rightTurn", "running"]
start = 3200
start = start / 20.0
tpr = 24
for r in np.arange(5):
    axt = layout.axes['ex_t{}'.format(r+1)]['axis']
    for behavior in order:
        max_neuron = best_neurons[behavior]
        axt.vlines(traces.loc[start:start+tpr, max_neuron].index,
                   0, traces.loc[start:start+tpr, max_neuron],
                   lw=.35, clip_on=False, color=style.getColor(behavior)) 
        
    for behavior in order:#+["stationary"]:
        axt.fill_between(traces.index.values, 11, -1,              
                         where=coords['behavior'] == behavior,
                         color=style.getColor(behavior), lw=0, alpha=.15)
    axt.set_ylim((-1,12))
    axt.set_xlim((start, start+tpr))
    axt.axis('off')
    start += tpr
y0=-2
sd=6
x1=start+.3
sec=2
axt.vlines(x1, y0, y0+sd, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
axt.text(x1+.25, y0+sd/2, '{}sd'.format(sd), ha='left', va='center',
         fontdict={'fontsize':6})
axt.hlines(y0, x1-sec, x1, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
axt.text(x1-sec/2, y0-1, '{}s'.format(sec), ha='center', va='top',
         fontdict={'fontsize':6})

axt = layout.axes['ex_t1']['axis']
patches = [mpatches.Patch(color=style.getColor(b), label=behaviorNames[b],
                          alpha=.15) for b in behaviorOrder]
axt.legend(handles=patches, ncol=4, mode='expand', bbox_to_anchor=(0,1.02,1,1.02),
           loc='lower center')

#%% Pie charts
df = tuningData.copy()

# only keep max tuning for each neuron
maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~df.signp, 'action'] = 'blank' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

for g in genotypeOrder:
    ax = layout.axes['pie_of_{}'.format(g)]['axis']
    gdata = maxdf.loc[g].loc[['leftTurn', 'rightTurn', 'blank', 'running', 'stationary']]
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[style.getColor(b) for b in gdata.index])

    ax.set_aspect('equal')

#%% Example tuning FOV
ax = layout.axes['tuning_fov_of']['axis']
df = ex_tunings.loc[ex_tunings.groupby('neuron').tuning.idxmax()]
colors = df.action.copy()
colors[~df.signp] = 'none'
colors = np.array([style.getColor(c) for c in colors])
rois = exampleSess.readROIs()
roiPlot.roiPlot(rois, colors, ax)
sel_cnts = np.array(list(rois[best_neurons[order]].idxmax(axis=0)))
ax.scatter(sel_cnts[:,1], sel_cnts[:,0], marker='o', edgecolor='k', facecolor='none', 
           s=8, alpha=1, lw=mpl.rcParams['axes.linewidth'])
ax.axis('off')

#%%
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
decodingData = analysisOpenField.decodeWithIncreasingNumberOfNeurons(endoDataPath)
decodingData.insert(1, "genotype", decodingData.session.str.split("_").str[0])

ax = layout.axes["decodeWithIncreasingNumberOfNeurons"]["axis"]
for strSess, df in decodingData.groupby("session"):
    genotype = strSess.split("_")[0]
    ax.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=0.2, lw=.5)
    ax.plot(df.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
             alpha=0.2, lw=.5)
for genotype, df in decodingData.groupby("genotype"):
    ax.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=1.0)
ax.plot(decodingData.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
         alpha=1.0)

meanHandles = [mpl.lines.Line2D([], [], color=style.getColor(g)) for g in genotypeOrder]
shuffleHandle = mpl.lines.Line2D([], [], color=style.getColor("shuffled"))
ax.legend(meanHandles+[shuffleHandle],
          [longGtNames[g] for g in genotypeOrder]+["shuffled"],
          loc=(0.2, 0.9), ncol=2, columnspacing=1.2)
ax.set_xlim(0,300)
ax.set_ylim(0,1)
ax.set_xlabel("number of neurons")
ax.set_ylabel("accuracy (%)")
ax.set_xticks(np.arange(50, 300, 50), minor=True)
ax.set_yticks(np.linspace(0,1,5))
ax.set_yticklabels(np.linspace(0,100,5,dtype=np.int64))
ax.set_title("SVM decoding of behavior", pad=12)
sns.despine(ax=ax)
#%%
tuningData = analysisTunings.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005

tuningData_shuffled = analysisTunings.getTuningData_shuffled(endoDataPath)
tuningData_shuffled['signp'] = tuningData_shuffled['pct'] > .995
tuningData_shuffled['signn'] = tuningData_shuffled['pct'] < .005

#%%
ex_session = ('oprm1','5308','190131')
s = next(readSessions.findSessions(endoDataPath, genotype=ex_session[0],
                                   animal=ex_session[1], date=ex_session[2],
                                   task='2choice'))
traces = s.readDeconvolvedTraces(rScore=True)
lfa = s.labelFrameActions(reward='sidePorts').set_index(traces.index)
actions = ['pC2L-','mC2R-','pL2Cr']
tunings = tuningData.query("genotype == @ex_session[0] & animal == @ex_session[1] & "+
                           "date == @ex_session[2]").copy()

sel_traces = {}
sel_neurons = []
for p,(a,t) in enumerate(tunings.query('action in @actions').groupby('action')):
    max_neuron = t.loc[t.tuning.idxmax(),'neuron']
    trace = traces[max_neuron]
    sel_traces[a] = trace
    sel_neurons.append(int(max_neuron))

start = 12.265*60
tpr = 24
for r in np.arange(5):
    labels = lfa.loc[start:start+tpr, ['label']]
    rewards = labels.loc[labels.label.str.endswith('r').astype('int').diff()==1].index.values
    axt = layout.axes['f8_t{}'.format(r+1)]['axis']
    
    for a,trace in sel_traces.items():
        axt.vlines(trace.loc[start:start+tpr].index,
                   0, trace.loc[start:start+tpr],
                   lw=.35, clip_on=False, color=style.getColor(a[:4])) 
        
    for l in ['pC','pR','pL']:
        axt.fill_between(labels.index.values, 11, -1,              
                         where=labels['label'].str.slice(0,2) == l,
                         color=style.getColor(l), lw=0, alpha=.15)
        
    for r in rewards:
        fancyViz.drawWaterDrop(axt, np.array([r, 9.7]), np.array([.31,1.5]),
                               facecolor='k')
        axt.axvline(r, .05, .67, lw=.5, ls='--', color='k')
    axt.set_ylim((-1,12))
    axt.set_xlim((start, start+tpr))
    axt.axis('off')
    start += tpr

y0=-2
sd=6
x1=start+.3
sec=2
axt.vlines(x1, y0, y0+sd, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
axt.text(x1+.25, y0+sd/2, '{}sd'.format(sd), ha='left', va='center',
         fontdict={'fontsize':6})
axt.hlines(y0, x1-sec, x1, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
axt.text(x1-sec/2, y0-1, '{}s'.format(sec), ha='center', va='top',
         fontdict={'fontsize':6})

axt = layout.axes['f8_t1']['axis']
patches = [mpatches.Patch(color=style.getColor(l), label=t, alpha=.15) 
               for l,t in [('pL','left port'),('pC','center port'),('pR','right Port')]]
axt.legend(handles=patches, ncol=3, mode='expand', bbox_to_anchor=(0,1.02,1,1.02),
           loc='lower center')

#%% map
ax = layout.axes['tuning_fov']['axis']
df = tunings.loc[tunings.groupby('neuron').tuning.idxmax()]
colors = df.action.copy()
colors[~df.signp] = 'none'
colors = np.array([style.getColor(c[:4]) for c in colors])
rois = s.readROIs()
roiPlot.roiPlot(rois, colors, ax)

sel_cnts = np.array(list(rois[sel_neurons].idxmax(axis=0)))
ax.scatter(sel_cnts[:,1], sel_cnts[:,0], marker='o', edgecolor='k', facecolor='none', 
           s=25, alpha=1, lw=mpl.rcParams['axes.linewidth'])
ax.axis('off')

#%%
df = tuningData.copy()

# only keep max tuning for each neuron
maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~maxdf.signp, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

# create dictionary with modified alpha to separate r/o/d phases
cdict = defaultdict(lambda: np.array([1,1,1]),
                    {a:style.getColor(a[:4]) for a 
                     in ['mC2L-','mC2R-','mL2C-','mR2C-','pC2L-','pC2R-','pL2C-','pR2C-']})
cdict['pL2Cr'] = cdict['pL2C-']
cdict['pL2Co'] = np.append(cdict['pL2C-'], .45)
cdict['dL2C-'] = np.append(cdict['pL2C-'], .7)
cdict['pR2Cr'] = cdict['pR2C-']
cdict['pR2Co'] = np.append(cdict['pR2C-'], .45)
cdict['dR2C-'] = np.append(cdict['pR2C-'], .7)
cdict['pC2L-'] = np.append(cdict['pC2L-'], .45)

for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']

    gdata = maxdf.loc[g].loc[['mC2R-','mL2C-','mC2L-','mR2C-','none',
                              'dL2C-','pL2Co','pL2Cr',
                              'dR2C-','pR2Co','pR2Cr',
                              'pC2L-','pC2R-',]]
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[cdict[a] for a in gdata.index])

    ax.set_aspect('equal')

#%%
phaseRasterData = analysisTunings.getPhaseRasterData(endoDataPath)
phaseRasterData.columns = phaseRasterData.columns.astype("float")
for (action, genotype), df in phaseRasterData.groupby(["action", "genotype"]):
    mean = df.mean(axis=0)
    sem = df.sem(axis=0)
    ax = layout.axes['psth_'+action[:4]]['axis']
    ax.fill_between(mean.index, mean-sem, mean+sem,
                    color=style.getColor(genotype), alpha=0.2)
    ax.plot(mean, color=style.getColor(genotype))
    if genotype == "oprm1":
        ax.set_xlim(0, 15)
        ax.set_ylim(-0.1, 0.5)
        ax.set_xticks([])
        if action == "mC2L-" or action == "mL2C-":
            ax.set_yticks((-0.1, 0.5))
            ax.set_yticks(np.arange(0, 0.5, 0.1), minor=True)
            ax.set_ylabel("z-score", labelpad=-7)
        else:
            ax.set_yticks([])
            ax.set_yticks(np.arange(-0.1, 0.51, 0.1), minor=True)
        if action =="mL2C-" or action == "mR2C-":
            ax.set_xlabel("scaled time")
        ax.axvspan(0, 5, color=style.getColor("p"+action[1]), alpha=0.2)
        ax.axvspan(10, 15, color=style.getColor("p"+action[3]), alpha=0.2)
        ax.axhline(0, ls=':', c='k', lw=0.5, alpha=0.5)
        #ax.axvspan(-0.5, 0, color=style.getColor(action[:4]), clip_on=False, zorder=-1)
        sns.despine(ax=ax)
        actionColor = style.getColor(action[:4])
        ax.tick_params(axis='y', colors=actionColor, which="minor")
        ax.spines['bottom'].set_color(actionColor)
        ax.spines['left'].set_color(actionColor)
genotypeOrder = ("d1", "a2a", "oprm1")
lines = [mpl.lines.Line2D([], [], color=style.getColor(gt)) for gt in genotypeOrder]
labels = ["D1", "A2A", "Oprm1"]
layout.axes["psth_mC2L"]["axis"].legend(lines, labels, ncol=3, columnspacing=1.2,
                                            bbox_to_anchor=(0.85, 1.2, 1, 0.1))

#%%
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
openFieldTunings = analysisOpenField.getTuningData(endoDataPath, segmentedBehavior)

twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)

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
order_twoChoice = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "dL2C-", "pL2Co", "pL2Cr",
                   "pC2L-", "pC2R-", "dR2C-", "pR2Co", "pR2Cr", "none"]
primaryPairs.action_2choice = pd.Categorical(primaryPairs.action_2choice, order_twoChoice)
primaryPairs.action_openField = pd.Categorical(primaryPairs.action_openField, order_openField)

colormap = {a: style.getColor(a[:4]) for a in primaryPairs.action_2choice.unique()}
colormap.update({a: style.getColor(a) for a in primaryPairs.action_openField.unique()})

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn', 'none': 'untuned'}
phaseNames = {'mC2L': 'center-to-left', 'mC2R': 'center-to-right', 'mL2C': 'left-to-center',
              'mR2C': 'right-to-center', 'pC': 'center port', 'pL2C': 'left port',
              'pR2C': 'right port', 'none': 'untuned'}

for gt in ("d1",):#, "a2a", "oprm1"):
    ax = layout.axes['alluvial_{}'.format(gt)]['axis']
    data = primaryPairs.query("genotype == '{}'".format(gt))
    leftY, rightY = alluvialPlot.alluvialPlot(data, "action_openField",
                                              "action_2choice", colormap, ax, 
                                              colorByRight=False, alpha=0.75)
    ax.set_xlim(-0.2,1.2)
    ax.axis("off")
    ax.set_title(genotypeNames[gt])
    if gt=="d1":
        ticks = leftY.groupby(level=0).agg({'lower': np.min, 'higher': np.max}).mean(axis=1)
        for label, y in ticks.items():
            ax.text(-0.25, y, behaviorNames[label], ha="right", va="center",
                    color=style.getColor(label))
    if gt=="d1":
        newLabels = rightY.index.get_level_values(1).str[:4]
        newLabels = newLabels.str.replace('pC2.', 'pC')
        newLabels = newLabels.str.replace('d', 'p')
        ticks = rightY.groupby(newLabels).agg({'lower': np.min, 'higher': np.max}).mean(axis=1)
        for label, y in ticks.items():
            ax.text(1.25, y, phaseNames[label], ha="left", va="center",
                    color=style.getColor(label))


#%%
cachedDataPaths = [cacheFolder / name for name in ['stsw_m.pkl','stsw_p.pkl',
                                                   'stsw_c.pkl']]
if np.all([path.is_file() for path in cachedDataPaths]):
    M = pd.read_pickle(cachedDataPaths[0])
    P = pd.read_pickle(cachedDataPaths[1])
    C = pd.read_pickle(cachedDataPaths[2])
else:
    M = pd.DataFrame() # confusion matrices (shuffle and real)
    P = pd.DataFrame() # action (probability) predictions
    C = pd.DataFrame() # svm coefficients
    
    for action in ['dL2C','mL2C','pC2L','mC2L','dR2C','mR2C','pC2R','mC2R']:
        (rm,rp,rc), (sm,sp,sc) = analysisStaySwitchDecoding.decodeStaySwitch(endoDataPath, action)
        
        for df in [rm,rp,rc,sm,sp,sc]:
            df.insert(0, 'action', action)
        
        m = pd.concat([rm,sm], axis=0, keys=[False,True], names=['shuffled']).reset_index('shuffled')
        M = M.append(m, ignore_index=True)
        
        p = pd.concat([rp,sp], axis=0, keys=[False,True], names=['shuffled']).reset_index('shuffled')
        P = P.append(p, ignore_index=True)
    
        c = pd.concat([rc,sc], axis=0, keys=[False,True], names=['shuffled']).reset_index('shuffled')
        C = C.append(c, ignore_index=True)
    
    M.to_pickle(cacheFolder / 'stsw_m.pkl')
    P.to_pickle(cacheFolder / 'stsw_p.pkl')
    C.to_pickle(cacheFolder / 'stsw_c.pkl')   

cachedDataPaths = [cacheFolder / name for name in ['logRegCoefficients.pkl',
                                                   'logRegDF.pkl']]
if np.all([path.is_file() for path in cachedDataPaths]):
    logRegCoef = pd.read_pickle(cachedDataPaths[0])
    logRegDF = pd.read_pickle(cachedDataPaths[1])
else:
    logRegCoef, logRegDF = analysisStaySwitchDecoding.getAVCoefficients(endoDataPath)
    logRegCoef.to_pickle(cachedDataPaths[0])
    logRegDF.to_pickle(cachedDataPaths[1])
    
cachedDataPath = cacheFolder / 'actionValues.pkl'
if cachedDataPath.is_file():
    actionValues = pd.read_pickle(cachedDataPath)
else:
    actionValues = analysisStaySwitchDecoding.getActionValues(endoDataPath, logRegCoef)
    actionValues.to_pickle(cachedDataPath)
    
cachedDataPath = cacheFolder / 'actionValues_shuffled.pkl'
if cachedDataPath.is_file():
    actionValues_shuffled = pd.read_pickle(cachedDataPath)
else:
    actionValues_shuffled = analysisStaySwitchDecoding.getActionValues(endoDataPath, logRegCoef,
                                                                       on_shuffled=True)
    actionValues_shuffled.to_pickle(cachedDataPath)
    
cachedDataPath = cacheFolder / 'staySwitchAUC.pkl'
if cachedDataPath.is_file():
    staySwitchAUC = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath,
                                                                  n_shuffles=1000)
    staySwitchAUC.to_pickle(cachedDataPath)
    
cachedDataPath = cacheFolder / 'staySwitchAUC_shuffled.pkl'
if cachedDataPath.is_file():
    staySwitchAUC_shuffled = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC_shuffled = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath,
                                                                           n_shuffles=1,
                                                                           on_shuffled=True)
    staySwitchAUC_shuffled.to_pickle(cachedDataPath)

#%% plot stay vs switch ROC AUC tuning distributions
#  could be done with "tuning" defined as z-scored by shuffle dist?
staySwitch = staySwitchAUC.loc[staySwitchAUC.action.isin(['mR2C'])].copy()
staySwitch_shuffled = staySwitchAUC_shuffled[staySwitchAUC_shuffled.action.isin(['mR2C'])].copy()
staySwitch_shuffled['genotype'] = 'shuffled'
staySwitch = pd.concat([staySwitch, staySwitch_shuffled])
palette = {gt: style.getColor(gt) for gt in ['d1','a2a','oprm1','shuffled']}

# plot kde
for a, adata in staySwitch.groupby('action'):
    ax = layout.axes['{}_auc_kde'.format(a)]['axis']
    
    for gt, agdata in adata.groupby('genotype'):
#        ax.hist(agdata['auc'], bins=np.arange(-1,1.1,.1), histtype='step',
#                color=style.getColor(gt), label=gt, density=True,
#                lw=2, alpha=.8)
        sns.distplot(agdata['auc'], bins=np.arange(-1,1.1,.1),
                     ax=ax, color=style.getColor(gt), hist=False,
                     kde_kws={'clip_on':False, 'alpha':.75,
                              'zorder':-1 if gt == 'shuffled' else 1})
     
    ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
    ax.set_ylim((0,4))
    ax.set_yticks((1,3), minor=True)
    ax.set_yticks((0,2,4))
    ax.set_yticklabels(())
    ax.set_ylabel('')
    if a == 'mR2C':
        ax.set_ylabel('density')
        ax.set_yticklabels(ax.get_yticks())
    ax.set_xlim((-1,1))
    #ax.set_xticks(())
    ax.set_xticks((-1,0,1))
    ax.set_xticklabels((-1,0,1))
    ax.set_xticks((-.5,.5), minor=True)
    ax.set_xlabel('selectivity score')
    sns.despine(bottom=False, trim=True, ax=ax)
    
axt = layout.axes['auc_legend']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.75,
                                 label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1',
                                        'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']
                  ]
axt.legend(handles=legend_elements, ncol=2, loc='center',
           mode='expand')
axt.axis('off')

#%% session bar plots showing fractions of stay-switch tuned neurons per session
action_aucs = staySwitchAUC.query('action == "mR2C"').copy()
action_aucs['stay'] = action_aucs.pct > .995
action_aucs['switch'] = action_aucs.pct < .005

sign_sess_frac = pd.DataFrame(action_aucs.groupby(['genotype','animal','date'])[['stay']].sum())
sign_sess_frac['switch'] = action_aucs.groupby(['genotype','animal','date']).switch.sum()
sign_sess_frac['noNeurons'] = action_aucs.groupby(['genotype','animal','date']).size()
sign_sess_frac.loc[:,['stay','switch']] = (sign_sess_frac[['stay','switch']] / 
                                           sign_sess_frac.noNeurons.values[:,np.newaxis])
sign_sess_frac.reset_index(inplace=True)

for tuning in ['stay','switch']:
    ax = layout.axes['perc_{}'.format(tuning)]
    sessionBarPlot.sessionBarPlot(sign_sess_frac, tuning, ax, style.getColor,
                                  weightScale=.0075)
    ax.axhline(0, ls=':', lw=mpl.rcParams['axes.linewidth'], color='k', alpha=.5,
               clip_on=False)
#    ax.axvline(0, ls=':', lw=mpl.rcParams['axes.linewidth'], color='k', alpha=.5,
#               clip_on=False)
    ax.set_ylim((0,.3))
    ax.set_yticks((0,.3))
    ax.set_yticks((.1,.2), minor=True)
    ax.set_yticklabels(())
    if tuning == "switch":
        ax.set_yticklabels((ax.get_yticks() * 100).astype('int'))
        ax.set_ylabel('% selective')
    ax.set_xticks(())
    ax.set_title('{}'.format({'stay':'win-stay','switch':'lose-switch'}[tuning]),
                 pad=4)
    sns.despine(ax=ax, trim=False, bottom=True)
#    if tuning == 'switch':
#        ax.invert_xaxis()


#%% pie charts
df = staySwitchAUC.copy()
df = df.query('action not in ["pL2C","pR2C"]')
df['sign'] = (df.pct > .995) | (df.pct < .005)

# only keep max tuning for each neuron
maxdf = (df.loc[df.groupby(['genotype','animal','date','neuron'])
                  .auc.apply(lambda t: t.abs().idxmax())])
# inidcate whether stay or switch tuned
maxdf.loc[maxdf.tuning.apply(np.sign) == 1, 'action'] += 'r.'
maxdf.loc[maxdf.tuning.apply(np.sign) == -1, 'action'] += 'o!'
maxdf.loc[~maxdf.sign, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['sign']].count() # get counts

# create dictionary with modified alpha to separate center port phases
cdict = {a:style.getColor(a[:4]) for a in maxdf.reset_index().action.unique()}
cdict['none'] = np.array((1,1,1))
cdict['pC2Lr.'] = np.append(cdict['pC2Lr.'], .45)
cdict['pC2Lo!'] = np.append(cdict['pC2Lo!'], .45)


for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_staySwitch_{}'.format(g)]['axis']

    order = ['mC2Ro!','mC2Rr.','mL2Co!','mL2Cr.',
             'mC2Lo!','mC2Lr.','mR2Co!','mR2Cr.',
             'none',
             'dL2Co!','dL2Cr.','dR2Co!','dR2Cr.',
             'pC2Lo!','pC2Lr.','pC2Ro!','pC2Rr.']
    gdata = maxdf.loc[g].loc[order]
    ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
           explode=[.1]*len(gdata), textprops={'color':'k'},
           colors=[cdict[a] for a in gdata.index])
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'k'},
                    explode=[.1]*len(gdata))
    for (w,a) in zip(ws,order):
        w.set_fill(False)
        if a.endswith('o!'):
            w.set_hatch('X'*10)
    
    ax.set_aspect('equal')

ax = layout.axes['pie_legend']['axis']
legend_elements = [mpatches.Wedge((0,0), 1, 80,100, edgecolor='k', facecolor='w', lw=.35,
                                  label='win-stay'),
                   mpatches.Wedge((0,0), 1, 80, 100, edgecolor='k', facecolor='w', lw=.35,
                                  hatch='X'*10, label='lose-switch')]
ax.legend(handles=legend_elements, loc='center')
ax.axis('off')


##%% tuning counts (simple)
hist_df = analysisStaySwitchDecoding.getTunedNoHistData(df)

axs = {}
for g, gdata in hist_df.query('bin != 0').groupby('genotype'):
    ax = layout.axes['no_tuned_'+g]['axis']
    axs[g] = ax
    
    ax.scatter(analysisStaySwitchDecoding.jitter(gdata.bin, .12), gdata.sign,
               s=gdata.noNeurons/25, edgecolor=style.getColor(g),
               facecolor='none', alpha=.8, zorder=0, clip_on=False,
               lw=mpl.rcParams['axes.linewidth'])
    
    avg = gdata.groupby('bin').apply(analysisStaySwitchDecoding.wAvg, 'sign', 'noNeurons')
    sem = gdata.groupby('bin').apply(analysisStaySwitchDecoding.bootstrap, 'sign', 'noNeurons')
    ax.bar(avg.index, avg, yerr=sem, color=style.getColor(g),
           lw=0, alpha=.3, zorder=1)
    
    ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])
    ax.set_ylim((0,.5))
    ax.set_yticks((0,.25,.5))
    ax.set_yticklabels(())
    ax.yaxis.set_minor_locator(MultipleLocator(.125))
    ax.set_xlim((0.25,5.75))
    ax.set_xticks((1,3,5))
    ax.xaxis.set_minor_locator(FixedLocator((2,4)))
    ax.set_xticklabels(['1','3','5+'])
    sns.despine(ax=ax)
    

axs['d1'].set_yticklabels((0,25,50))
axs['d1'].set_ylabel('neurons (%)')
axs['a2a'].set_xlabel('number of phases')

#%% stay-switch decoding accuracy versus shuffled & action duration decoders
acc = P.loc[P.label.str.contains('r\.$|o!$')].copy() # only use win-stay, lose-switch trials
acc_groupby = acc.groupby(['genotype','animal','date','noNeurons','action', 'shuffled'])
acc_activity = acc_groupby.apply(lambda sess: np.mean(sess.prediction == sess.label))
acc_activity.name = 'accuracy'
acc_speed = acc_groupby.apply(lambda sess: np.mean(sess.duration_prediction == sess.label))
acc_speed.name = 'accuracy'
acc = pd.concat([acc_activity, acc_speed], keys=['activity','speed'], names=['decoder'])
acc = acc.reset_index('noNeurons')
acc = acc.reorder_levels((0,5,1,2,3,4))
   
##%%
for (gt, a), gdata in acc.groupby(['genotype','action']):
    ax = layout.axes['{}_{}'.format(gt, a)]['axis']
    
    wAvgs = (gdata.groupby(['decoder','shuffled'])
                  .apply(analysisStaySwitchDecoding.wAvg, 'accuracy', 'noNeurons'))
    wSems = (gdata.groupby(['decoder','shuffled'])
                  .apply(analysisStaySwitchDecoding.bootstrap, 'accuracy', 'noNeurons'))
    
    decs = [('activity',True), ('activity',False), ('speed',False)]
    for x, (dec,shuffle) in enumerate(decs):
        ax.errorbar(x, wAvgs[dec,shuffle], yerr=wSems[dec,shuffle],
                    color=style.getColor(a), clip_on=False,
                    marker={0:'o',1:'v',2:'s'}[x],
                    markersize={0:3.2,1:3.6,2:2.8}[x],
                    markerfacecolor='w',
                    markeredgewidth=.8)
   
    ax.plot([0,1,2], [wAvgs.loc[dec,shuffle] for dec,shuffle in decs],
            color=style.getColor(a), clip_on=False)
    
    for s, sdata in gdata.groupby(['animal','date']):
        ax.plot([0,1,2], [sdata.loc[dec,shuffle].accuracy for dec,shuffle in decs],
                color=style.getColor(a), alpha=.2,zorder=-99,
                lw=.5, clip_on=False)
    
    ax.axhline(0.5, lw=mpl.rcParams['axes.linewidth'], c='k', alpha=.5, ls=':', clip_on=False)
    
    ax.set_ylim((.5,1))
    ax.set_xlim((-.35,2.35))
    ax.set_xticks(())
    ax.set_yticklabels(())
    ax.axis('off')
    if a == 'mL2C':
        ax.axis('on')
        ax.set_yticks((.5,.75,1))
        if gt == 'a2a':
            ax.set_yticklabels((50,75,100))
            ax.set_ylabel('decoder accuracy (%)')
    else:
        ax.set_yticklabels(())
    
    sns.despine(ax=ax, bottom=True, offset=.5)

#%% svm prediction plots
prob_value_df = (P.set_index(['shuffled','genotype','animal','date','label','actionNo'])
                  .loc[False, ['action','o!','r.','noNeurons']])
prob_value_df['value'] = (actionValues.set_index(['genotype','animal','date','label','actionNo'])
                                      .value)
prob_value_df = prob_value_df.reset_index()
prob_value_df['stay'] = prob_value_df.label.str.endswith('.').astype('int')
    
##%%
data = prob_value_df.query('action in ["mL2C","mR2C"]').dropna().copy()
data = data.loc[data.label.str.contains('o!$|o\.$|r\.$')]

for (gt,label), gdata in data.groupby(['genotype','action']):
    ax = layout.axes['{}_value_ost'.format(gt)]['axis']
    
    for tt in ['o!','r.','o.']:
        ttdata = gdata.loc[gdata.label.str.endswith(tt)].copy()
        ttdata['bin'] = pd.qcut(ttdata.value, 4).cat.codes
        ttdata = ttdata.groupby(['animal','date','bin'])[['noNeurons','value','r.']].mean()
        
        stsw_wAvg = (ttdata.groupby('bin')
                           .apply(analysisStaySwitchDecoding.wAvg,'r.','noNeurons'))
        stsw_wSem = (ttdata.groupby('bin')
                           .apply(analysisStaySwitchDecoding.bootstrap,'r.','noNeurons'))
        value_wAvg = (ttdata.groupby('bin')
                            .apply(analysisStaySwitchDecoding.wAvg,'value','noNeurons'))
        value_wSem = (ttdata.groupby('bin')
                            .apply(analysisStaySwitchDecoding.bootstrap,'value','noNeurons'))
        
        ax.errorbar(value_wAvg, stsw_wAvg, xerr=value_wSem, yerr=stsw_wSem,
                    color=style.getColor(tt),
                    lw=0, marker='>' if 'R' in label else '<',
                    markersize=2.8, clip_on=False, barsabove=False,
                    alpha=1, markeredgewidth=0, elinewidth=.5)
        ax.fill_between(value_wAvg, stsw_wAvg-stsw_wSem, stsw_wAvg+stsw_wSem,
                        lw=0, alpha=.35, zorder=-1, color=style.getColor(tt))
    
    ax.axhline(.5, ls=':', c='k', alpha=.35, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
    ax.axvline(0, ls=':', c='k', alpha=.35, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
    
    ax.set_ylim((0,1))
    ax.set_xlim((-5,5))
    ax.set_xticks((-5,0,5))
    #ax.invert_xaxis()
    if gt == 'a2a':
        ax.set_xlabel('action value')
    ax.set_yticks((0,.5,1))
    if gt == 'd1':
        ax.set_yticklabels((0,50,100))
        #ax.set_yticklabels((-100, 0, 100))
        #ax.set_ylabel('SVM prediction\nP(win-stay)')
        #ax.set_ylabel('certainty')
    else:
        ax.set_yticklabels(())
    ax.yaxis.set_minor_locator(MultipleLocator(.25))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    sns.despine(ax=ax)


for gt, gdata in data.groupby('genotype'):
    axkde = layout.axes['{}_value_kde'.format(gt)]['axis']

    gdata = gdata.copy()
    gdata['tt'] = gdata.label.str.slice(-2)
    gdata = gdata.set_index(['animal','date'])
    
    bins = np.arange(-5.5, 5.6, .5)
    labels = (np.arange(-5.5, 5.6, .5) +.25)[:-1]
    gdata['bin'] = pd.cut(gdata.value, bins=bins, labels=labels).astype('float')
    gdist = gdata.groupby(['animal','date','tt','bin']).size().reset_index(['tt','bin'])
    gdist = gdist.rename(columns={0:'pct'})
    gdist['pct'] /= gdata.groupby(['animal','date']).size()
    gdist['noNeurons'] = gdata.groupby(['animal','date']).noNeurons.first()
    
    gdist_stats = gdist.groupby(['tt','bin']).pct.agg(['mean','sem']).reset_index('bin')
        
    for tt, ttdata in gdist_stats.groupby('tt'):
        axkde.plot(ttdata.bin, ttdata['mean'], color=style.getColor(tt),
                   lw=.5, clip_on=False)
        axkde.fill_between(ttdata.bin,
                           ttdata['mean']-ttdata['sem'], ttdata['mean']+ttdata['sem'],
                           color=style.getColor(tt), alpha=.35, lw=0,
                           clip_on=False)
    
    axkde.axvline(0, ls=':', c='k', alpha=.35, zorder=-1,
                  lw=mpl.rcParams['axes.linewidth'])
    
    axkde.set_xlim((-5,5))
    axkde.set_xticks(())
    axkde.set_ylim((0,.1))
    axkde.set_yticks((0,.05,.1))
    axkde.set_yticklabels(())
    if gt == 'd1':
        axkde.set_ylabel('% trials')
        axkde.set_yticklabels((0,5,10))
    sns.despine(bottom=True, trim=True, ax=axkde)
    axkde.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[gt])


axt = layout.axes['value_ost_legend']['axis']
legend_elements = [mlines.Line2D([0], [0], marker='<', color='k', markersize=2.8,
                                 markeredgewidth=0, label='(left) choice', lw=0),
                   mpatches.Patch(color=style.getColor('r.'), alpha=1,
                                  label='win-stay'),                          
                   mpatches.Patch(color=style.getColor('o.'), alpha=1,
                                  label='lose-stay'),
                   mpatches.Patch(color=style.getColor('o!'), alpha=1,
                                  label='lose-switch'),
                  ]
axt.legend(handles=legend_elements, ncol=len(legend_elements), loc='center',
           mode='expand')
axt.axis('off')


##%% svm prediciton X value correlations
def getCorr(ttdata):
    corr = ttdata.groupby(['genotype','animal','date'])[['r.','value']].corr()
    corr = pd.DataFrame(pd.Series(corr.unstack(-1)[('value','r.')],
                                  name='correlation'))
    corr['noNeurons'] = ttdata.groupby(['genotype','animal','date']).noNeurons.first()
    return corr

def randomShiftValue(ttdata):
    def shift(v):
        v = pd.Series(np.roll(v, np.random.randint(10,30) * np.random.choice([-1,1])),
                      index=v.index)
        return v
    
    ttdata = ttdata.copy()
    ttdata['value'] = ttdata.groupby(['genotype','animal','date'])['value'].apply(shift).copy()
    return ttdata
    

valueProbCorrs = pd.DataFrame()
for tt, ttdata in data.groupby(data.label.str.slice(-2)):
    ttdata = ttdata.copy()
    left_trials = ttdata.label.str.contains('L')
    ttdata.loc[left_trials, 'value'] = ttdata.loc[left_trials, 'value'] * -1
    
    corr = getCorr(ttdata)
    
    #ttdata['absValue'] = np.random.permutation(ttdata.absValue)
    ttdata_vshifted = randomShiftValue(ttdata)
    r_corr = getCorr(ttdata_vshifted)

    corr['rand_correlation'] = r_corr['correlation']
    corr['trialType'] = tt
    corr = corr.set_index('trialType', append=True)
    
    valueProbCorrs = valueProbCorrs.append(corr)


for (gt,tt), cs in (valueProbCorrs.query('trialType in ["r.","o.","o!"]')
                                  .groupby(['genotype','trialType'])):
    ax = layout.axes['{}_{}_corr'.format(gt,tt)]['axis']
    
    wAvg = analysisStaySwitchDecoding.wAvg(cs, 'correlation', 'noNeurons')
    wSem = analysisStaySwitchDecoding.bootstrap(cs, 'correlation', 'noNeurons')
    r_wAvg = analysisStaySwitchDecoding.wAvg(cs, 'rand_correlation', 'noNeurons')
    r_wSem = analysisStaySwitchDecoding.bootstrap(cs, 'rand_correlation', 'noNeurons')
    
#    ax.bar([0,1], [wAvg, r_wAvg], yerr=[wSem, r_wSem],
#           color=[style.getColor(tt), style.getColor('shuffled')],
#           lw=0, alpha=.5, zorder=1, width=.5)
    ax.errorbar(0, wAvg, yerr=wSem, color=style.getColor(tt), clip_on=False,
                marker='v', markersize=3.6, markerfacecolor='w',
                markeredgewidth=.8)
    ax.errorbar(1, r_wAvg, yerr=r_wSem, color=style.getColor(tt), clip_on=False,
                marker='o', markersize=3.2, markerfacecolor='w',
                markeredgewidth=.8)
    ax.plot([0,1], [wAvg, r_wAvg], color=style.getColor(tt), clip_on=False)
    
    for c in cs[['correlation','rand_correlation','noNeurons']].values:
        ax.plot([0,1], c[:2], lw=mpl.rcParams['axes.linewidth'], alpha=.2,
                clip_on=False, zorder=-99, color=style.getColor(tt))
    
    ax.axhline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])

    ax.set_ylim((0,.5))    
    ax.set_xlim((-.35,1.35))
    if tt == 'r.':
        ax.set_xticks(())
        ax.set_yticks((0,.5))
        ax.set_yticklabels(())
        ax.set_yticks((.25,), minor=True)
        if gt == 'a2a':
            ax.set_ylabel('r(action value*, P(win-stay))')
            ax.set_yticklabels((.0,.5))
        sns.despine(ax=ax, bottom=True, trim=True)
    else:
        ax.set_axis_off()
 
    
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "paletteExample.svg")