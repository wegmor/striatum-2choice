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
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FixedLocator
import scipy.stats
from utils import readSessions, fancyViz, alluvialPlot, roiPlot
from collections import defaultdict
import pathlib
import figurefirst
import style
import analysisTunings
import analysisDecoding
import analysisStaySwitchDecoding
plt.ioff()

#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "figure3Tunings.svg")
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

#%% Panel J
decodingData = analysisDecoding.decodeWithIncreasingNumberOfNeurons(endoDataPath)
decodingData.insert(1, "genotype", decodingData.session.str.split("_").str[0])

##%%
plt.sca(layout.axes["decodeWithIncreasingNumberOfNeurons"]["axis"])
for strSess, df in decodingData.groupby("session"):
    genotype = strSess.split("_")[0]
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=0.35, lw=.35)
    plt.plot(df.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
             alpha=0.35, lw=.35)
for genotype, df in decodingData.groupby("genotype"):
    gavg = df.groupby('nNeurons').realAccuracy.mean()
    gsem = df.groupby('nNeurons').realAccuracy.sem()
    plt.plot(gavg, color=style.getColor(genotype), alpha=1.0)
    plt.fill_between(gavg.index, gavg-gsem, gavg+gsem, lw=0,
                     color=style.getColor(genotype), alpha=.2, zorder=-99)
savg = decodingData.groupby('nNeurons').shuffledAccuracy.mean()
ssem = decodingData.groupby('nNeurons').shuffledAccuracy.sem()
plt.plot(savg, color=style.getColor("shuffled"), alpha=1.0)
plt.fill_between(savg.index, savg-ssem, savg+ssem, lw=0,
                 color=style.getColor('shuffled'), alpha=.2, zorder=-99)

order = ("d1", "a2a", "oprm1")
genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
meanHandles = [mpl.lines.Line2D([], [], color=style.getColor(g), 
                                label=genotypeNames[g])
                   for g in order]
shuffleHandle = mpl.lines.Line2D([], [], color=style.getColor("shuffled"),
                                 label='shuffled')
plt.legend(handles=meanHandles+[shuffleHandle],
           bbox_to_anchor=(1,.28), loc='center right', ncol=2)

plt.ylim(0,1)
plt.xlim(0,200)
plt.xlabel("number of neurons")
plt.ylabel("decoding accuracy (%)")
plt.yticks((0,.5,1), (0,50,100))
plt.gca().set_yticks(np.arange(.25,1,.25), minor=True)
plt.xticks(np.arange(0,201,50))
plt.gca().set_xticks(np.arange(25,200,25), minor=True)
sns.despine(ax=plt.gca())


#%% Panel K
decodingData = analysisDecoding.decodingConfusion(endoDataPath)
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]

cmap = mpl.cm.RdYlGn
for gt, data in decodingData.groupby("genotype"):
    weightedData = data.set_index(["true", "predicted"]).eval("occurences * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    gtMeans = np.diag(weightedData)
    labels = [(l[:4] if l[-1]=='-' else l) for l in weightedData.columns]
    di = {k: cmap(v) for k, v in zip(labels, gtMeans)}
    plt.sca(layout.axes["decodingAccuracyPerLabel_{}".format(gt)]["axis"])
    fancyViz.drawBinnedSchematicPlot(di, lw=mpl.rcParams['axes.linewidth'])

cax = layout.axes["decodingAccuracyCbar"]
cb1 = mpl.colorbar.ColorbarBase(cmap=cmap, ax=cax, norm=mpl.colors.Normalize(vmin=0, vmax=100),
                                orientation='horizontal', ticks=(0,50,100))
cb1.outline.set_visible(False)
cax.set_axis_off()
cax.text(-.025, .25, 0, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, 100, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.125, 'recall (%)', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)

#%% Panel L
exampleNeurons = (7, 66, 13)
saturation = 1

sess = next(readSessions.findSessions(endoDataPath, animal="5308", date="190131"))
lfa = sess.labelFrameActions(reward="sidePorts")
deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
X = deconv[lfa.label=="mR2C-"]
Y = lfa.actionProgress[lfa.label=="mR2C-"]
avgActivity = X.groupby((Y*10).astype("int")/10.0).mean().T
sorting = avgActivity.idxmax(axis=1).argsort()
plt.sca(layout.axes["movementProgressRaster"]["axis"])
plt.imshow(avgActivity.iloc[sorting], aspect="auto",
           interpolation="nearest", vmin=-saturation, vmax=saturation, cmap="RdYlBu_r")
for i, y in enumerate(np.argsort(sorting)[list(exampleNeurons)]):
    plt.plot([13, 10, 13], [i*110+15, y, (i+1)*110-10], 'k', lw=0.5, ls=':', clip_on=False)
plt.xlim((-.5,9.5))
plt.ylim(len(sorting)-1, 0)
plt.xticks([])
plt.yticks([0, len(sorting)-1], [len(sorting), 1])
plt.xlabel("scaled\ntime", labelpad=3)
plt.ylabel('neuron (sorted)', labelpad=-8)
plt.title("right to\ncenter turn", pad=5)
plt.gca().tick_params(axis='both', which='both',length=0)
#plt.gca().spines['bottom'].set_color(style.getColor("mR2C"))
sns.despine(ax=plt.gca(), top=True, bottom=False, left=True, right=True)

fv = fancyViz.SchematicIntensityPlot(sess, linewidth=mpl.rcParams['axes.linewidth'],
                                     splitReturns=False, smoothing=7, saturation=saturation)
for i in range(3):
    ax = layout.axes["movementExample{}".format(i+1)]["axis"]
    img = fv.draw(deconv[exampleNeurons[i]], ax=ax)
    
cax = layout.axes['colorbar_turnDecoding']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-.025, .25, -saturation, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, saturation, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)

    
#%% Panel M
decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath)
decodingMovementProgress['genotype'] = decodingMovementProgress.sess.str.split('_').str[0]
decodingMovementProgress['animal'] = decodingMovementProgress.sess.str.split('_').str[1]
decodingMovementProgress['date'] = decodingMovementProgress.sess.str.split('_').str[2]
def calcCorr(df):
    r = scipy.stats.pearsonr(df.true, df.predicted)[0]
    return pd.Series((r, df.nNeurons.iloc[0]), ("correlation", "nNeurons"))

##%%
exampleSession = decodingMovementProgress.query("sess == 'oprm1_5308_190131' & not shuffle")
means = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.mean()
#xmeans = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).true.mean()
stds = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.std()
plt.sca(layout.axes["decodingProgressExample"]["axis"])
plt.plot([0,1], [0, 1], color='k', ls=':', alpha=0.5, lw=mpl.rcParams['axes.linewidth'])
plt.errorbar(means.index, means, yerr=stds, fmt='o-', ms=2.8,
             color=style.getColor("oprm1"), markeredgewidth=0)
plt.xlim(-0.05,1.00)
plt.ylim(-0.05,1.00)
plt.xticks((0,.5,1),())#, rotation=30, ha="right", va="top")
plt.yticks((0,.5,1))
plt.gca().set_xticks((.25,.75), minor=True)
plt.gca().set_yticks((.25,.75), minor=True)
corr = calcCorr(exampleSession).loc["correlation"]
plt.text(1, 0.015, "r = {:.3f}".format(corr), fontsize=6,
         color="k", ha="right", va='center')
plt.title("right to center\nturn")
sns.despine(ax=plt.gca())

##%%
decodingMovementProgress['bin'] = np.floor(decodingMovementProgress.true * 10)/10 +.05
moveProgAnimalMean = (decodingMovementProgress.groupby(['shuffle',
                                                        'genotype','animal','date',
                                                        'nNeurons','bin'])[['true','predicted']]
                                              .mean()
                                              .reset_index('nNeurons')
                                              .unstack('shuffle'))
moveProgAnimalMean.columns = moveProgAnimalMean.columns.reorder_levels((1,0))

plt.sca(layout.axes["decodingProgressAvg"]["axis"])
for gt, gdata in moveProgAnimalMean.groupby('genotype'):
    wAvg = gdata[False].groupby('bin').apply(analysisStaySwitchDecoding.wAvg,
                                             'predicted', 'nNeurons')
    wX = gdata[False].groupby('bin').apply(analysisStaySwitchDecoding.wAvg,
                                           'true', 'nNeurons')
    wSem = gdata[False].groupby('bin').apply(analysisStaySwitchDecoding.bootstrap,
                                             'predicted', 'nNeurons')
    
    plt.fill_between(wX, wAvg-wSem, wAvg+wSem,
                     lw=0, alpha=.35, zorder=-1, color=style.getColor(gt))
    plt.plot(wX, wAvg, color=style.getColor(gt),
             alpha=.8)
    
r_wAvg = moveProgAnimalMean[True].groupby('bin').apply(analysisStaySwitchDecoding.wAvg,
                                                       'predicted', 'nNeurons')
r_wX = moveProgAnimalMean[True].groupby('bin').apply(analysisStaySwitchDecoding.wAvg,
                                                     'true', 'nNeurons')
r_wSem = moveProgAnimalMean[True].groupby('bin').apply(analysisStaySwitchDecoding.bootstrap,
                                                       'predicted', 'nNeurons')

plt.fill_between(r_wX, r_wAvg-r_wSem, r_wAvg+r_wSem,
                 lw=0, alpha=.35, zorder=-2, color=style.getColor('shuffled'))
plt.plot(r_wX, r_wAvg, color=style.getColor('shuffled'), alpha=.8)

plt.plot([0,1], [0,1], color='k', ls=':', alpha=0.5, lw=mpl.rcParams['axes.linewidth'])
plt.xlim(-.05,1)
plt.ylim(-.05,1)
plt.xticks((0,.5,1))#, (0,50,100))
plt.gca().set_xticks((.25,.75), minor=True)
plt.yticks((0,.5,1))#, (0,50,100))
plt.gca().set_yticks((.25,.75), minor=True)
plt.xlabel("true scaled\ntime")
#plt.ylabel("predicted", labelpad=-2.25)
sns.despine(ax=plt.gca())


#%% Panel N
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
        ax.set_ylabel('r(true, predicted)')
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
layout.write_svg(outputFolder / "figure3Tunings.svg")
