#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:25:34 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FixedLocator
from utils import readSessions, fancyViz, alluvialPlot, roiPlot
from collections import defaultdict
import pathlib
import figurefirst
import style
import analysisTunings
import analysisOpenField
plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "tunings.svg")
layout.make_mplfigures()

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
    
    axfv = layout.axes['f8_{}'.format(p+1)]['axis']
    fv = fancyViz.SchematicIntensityPlot(s, splitReturns=False,
                                         linewidth=mpl.rcParams['axes.linewidth'],
                                         smoothing=7)
    img = fv.draw(trace, ax=axfv)
    
    axbg = layout.axes['f8_{}_bg'.format(p+1)]['axis']
    axbg.axvspan(-.055, -.03, .1, .93, color=style.getColor(a[:4]), alpha=1,
                 clip_on=False)
    axbg.set_xlim((0,1))
    axbg.set_axis_off()
    
cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
#cax.xaxis.tick_top()
#cax.tick_params(axis='both', which='both',length=0)
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})

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
                               facecolor='k')#sns.color_palette()[0])
        axt.axvline(r, .05, .67, lw=.5, ls='--', color='k')#sns.color_palette()[0])
        #axt.text(r, 10, '*', ha='center', va='top', fontdict={'fontsize':9})
        
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
'''
ax = layout.axes['tuning_fov']['axis']

df = tunings.loc[tunings.groupby('neuron').tuning.idxmax()].copy()
df['color'] = df.action
df.loc[~df.signp, 'color'] = 'none'
df['color'] = df.color.str.slice(0,4).apply(lambda c: np.array(style.getColor(c)))

rois = s.readROIs()
sel_cnts = np.array(rois.idxmax(axis=0).loc[sel_neurons].tolist())[:,::-1]
rois = np.array([rois[n].unstack('x').values for n in rois])

rs = []
for roi, color in zip(rois, df.color.values):
    roi /= roi.max()
    roi = roi**1.5
    roi = np.clip(roi-.1, 0, .85)
    roi /= roi.max()
    r = np.array([(roi > 0).astype('int')]*3) * color[:, np.newaxis, np.newaxis]
    r = np.concatenate([r, roi[np.newaxis]], axis=0)
    rs.append(r.transpose((1,2,0)))    
rs = np.array(rs)

for img in rs:
    ax.imshow(img)
ax.scatter(sel_cnts[:,0], sel_cnts[:,1], marker='o', edgecolor='k', facecolor='none', 
           s=25, alpha=1, lw=mpl.rcParams['axes.linewidth'])

ax.axis('off')
'''

#%%
ax = layout.axes['tuning_hist1']['axis']
hdata = tuningData.query('genotype == "oprm1" & action == "mC2L-"').copy()
shuffle_kde = tuningData_shuffled.query('genotype == "oprm1" & action == "mC2L-"').copy()

sns.kdeplot(shuffle_kde['tuning'], ax=ax, color=style.getColor('shuffled'), alpha=.75,
            clip_on=False, zorder=10, label='')
sns.kdeplot(hdata['tuning'], ax=ax, color='gray', alpha=.75, clip_on=True,
            zorder=-99, label='')
bins = np.arange(-20.5, 41.5)
none_hist = np.histogram(hdata.loc[~hdata['signp'], 'tuning'], bins=bins)[0] / len(hdata.tuning)
sign_hist = np.histogram(hdata.loc[hdata['signp'], 'tuning'], bins=bins)[0] / len(hdata.tuning)
#ax.hist(hdata['tuning'], bins=bins, lw=0, color='gray', alpha=.6,
#        histtype='stepfilled', align='mid')
ax.bar((bins+.5)[:-1], none_hist, lw=0, color='gray', alpha=.6)
#ax.hist(hdata.loc[hdata.signp,'tuning'], bins=bins, lw=0,
#        histtype='stepfilled', color=style.getColor('mC2L'), align='mid')
ax.bar((bins+.5)[:-1], sign_hist, lw=0, color=style.getColor('mC2L'), bottom=none_hist)

#ax.text(30,45,'significant\ntuning',ha='right',va='bottom',fontdict={'fontsize':7},
#        color=style.getColor('mC2L'))
ax.text(30,.04,'significant\npos. tuning',ha='right',va='bottom',fontdict={'fontsize':7},
        color=style.getColor('mC2L'))
#ax.text(7.5,400,'center to left\nturn',ha='center',va='center',fontdict={'fontsize':7})
ax.text(7.5,.45,'center to left turn',ha='center',va='center',
        fontdict={'fontsize':7})
ax.text(4.5,.25,'shuffled',ha='left',va='center',
        fontdict={'fontsize':7,'color':style.getColor('shuffled'),'alpha':.75})

#ax.set_yticks((0,200,400))
ax.set_yticks((.0,.2,.4))
#ax.yaxis.set_minor_locator(MultipleLocator(100))
ax.set_yticks((.1,.3), minor=True)
ax.set_xticks((-15,0,15,30))
ax.set_xlim((-15,30))
#ax.set_ylim((0,400))
ax.set_ylim((0,.42))
ax.set_xlabel('tuning score')
#ax.set_ylabel('# neurons')
ax.set_ylabel('density')
sns.despine(ax=ax, trim=True)


#%% pie charts
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
        ax.axhspan(0.5, 0.53, color=style.getColor(action[:4]), clip_on=False)
        sns.despine(ax=ax)
genotypeOrder = ("d1", "a2a", "oprm1")
lines = [mpl.lines.Line2D([], [], color=style.getColor(gt)) for gt in genotypeOrder]
labels = ["D1", "A2A", "Oprm1"]
layout.axes["psth_mC2L"]["axis"].legend(lines, labels, ncol=3, columnspacing=1.2,
                                            bbox_to_anchor=(0.85, 1.2, 1, 0.1)) 
#%% tuning counts (simple)
hist_df = analysisTunings.getTunedNoHistData(tuningData)

axs = {}
for g, gdata in hist_df.query('bin != 0').groupby('genotype'):
    ax = layout.axes['no_tuned_'+g]['axis']
    axs[g] = ax
    
    ax.scatter(analysisTunings.jitter(gdata.bin, .12), gdata.signp,
               s=gdata.noNeurons/25, edgecolor=style.getColor(g),
               facecolor='none', alpha=.8, zorder=0, clip_on=False,
               lw=mpl.rcParams['axes.linewidth'])
    
    avg = gdata.groupby('bin').apply(analysisTunings.wAvg, 'signp', 'noNeurons')
    sem = gdata.groupby('bin').apply(analysisTunings.bootstrap, 'signp', 'noNeurons')
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

#%% similar tuning == closer spatially?
pdists = analysisTunings.getPDistData(endoDataPath, tuningData)
    
#%%
ax = layout.axes['dist_scatter']['axis']

for g, gdata in pdists.groupby('genotype'):
    ax.scatter(gdata.dist_shuffle, gdata.dist, s=gdata.noNeurons/25,
               edgecolor=style.getColor(g), facecolor=style.getColor(g),
               alpha=.4, lw=mpl.rcParams['axes.linewidth'])
    
avg = pdists.groupby('genotype').apply(analysisTunings.wAvg, 'dist', 'noNeurons')
avg_s = pdists.groupby('genotype').apply(analysisTunings.wAvg, 'dist_shuffle', 'noNeurons')
sem = pdists.groupby('genotype').apply(analysisTunings.bootstrap, 'dist', 'noNeurons')
sem_s = pdists.groupby('genotype').apply(analysisTunings.bootstrap, 'dist_shuffle', 'noNeurons')

for g in ['d1','a2a','oprm1']:
    ax.errorbar(avg_s[g], avg[g], xerr=sem_s[g], yerr=sem[g],
                color=style.getColor(g), fmt='s', markersize=3,
                markeredgewidth=mpl.rcParams['axes.linewidth'],
                markeredgecolor='k', ecolor='k',
                label={'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])

ax.plot([25,75],[25,75], ls=':', color='k', alpha=.5, zorder=-1)    

ax.set_xlim((25,75))
ax.set_ylim((25,75))
ax.set_xticks(np.arange(25,76,25))
ax.set_yticks(np.arange(25,76,25))
ax.set_aspect('equal')
ax.set_xlabel('expected')
ax.set_ylabel('observed')
ax.text(50, 75, 'μm to nearest\ntuned neighbor', ha='center', va='center',
        fontdict={'fontsize':7})
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-3:], labels[-3:], loc='lower right', bbox_to_anchor=(1.1, .05))
ax.set_aspect('equal')
sns.despine(ax=ax)


#%%
genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}

## Panel J
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
#segmentedBehavior = segmentedBehavior.set_index("session")
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

for gt in ("d1", "a2a", "oprm1"):
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
    if gt=="oprm1":
        newLabels = rightY.index.get_level_values(1).str[:4]
        newLabels = newLabels.str.replace('pC2.', 'pC')
        newLabels = newLabels.str.replace('d', 'p')
        ticks = rightY.groupby(newLabels).agg({'lower': np.min, 'higher': np.max}).mean(axis=1)
        for label, y in ticks.items():
            ax.text(1.25, y, phaseNames[label], ha="left", va="center",
                    color=style.getColor(label))

## Panel K
examples = [
    ('d1_5643_190201', 112),
    ('a2a_5693_190131', 197),
    ('oprm1_5308_190201', 86)
]
for i in range(3):
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
    openFieldAx.set_title(genotypeNames[genotype] + " example", loc="left", pad=-4)
cax = layout.axes['second_colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "tunings.svg")
