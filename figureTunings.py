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
from utils import readSessions, fancyViz
from collections import defaultdict
import pathlib
import figurefirst
import style
import analysisTunings
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
layout = figurefirst.FigureLayout(templateFolder / "tunings.svg")
layout.make_mplfigures()


#%%
cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisTunings.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)

tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005

#%%
ex_session = ('oprm1','5308','190131')
s = next(readSessions.findSessions(endoDataPath, genotype=ex_session[0],
                                   animal=ex_session[1], date=ex_session[2],
                                   task='2choice'))
traces = s.readDeconvolvedTraces(zScore=True)
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

df = tunings.loc[tunings.groupby('neuron').tuning.idxmax()].copy()
df['color'] = df.action
df.loc[~df.signp, 'color'] = 'none'
df['color'] = df.color.str.slice(0,4).apply(lambda c: np.array(style.getColor(c)))

rois = s.readROIs().values
sel_cnts = analysisTunings.get_centers(rois)[sel_neurons]

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


#%%
ax = layout.axes['tuning_hist1']['axis']
hdata = tuningData.query('genotype == "oprm1" & action == "mC2L-"').copy()

ax.hist(hdata['tuning'], bins=np.arange(-20,40,1), lw=0, color='gray', alpha=.6,
        histtype='stepfilled')
ax.hist(hdata.loc[hdata.signp,'tuning'], np.arange(-20,40,1), lw=0,
        histtype='stepfilled', color=style.getColor('mC2L'))

ax.text(30,45,'significant\ntuning',ha='right',va='bottom',fontdict={'fontsize':7},
        color=style.getColor('mC2L'))
ax.text(7.5,400,'center to left\nturn',ha='center',va='center',fontdict={'fontsize':7})

ax.set_yticks((0,200,400))
ax.yaxis.set_minor_locator(MultipleLocator(100))
ax.set_xticks((-15,0,15,30))
ax.set_xlim((-15,30))
ax.set_ylim((0,400))
ax.set_xlabel('tuning score')
ax.set_ylabel('# neurons')
sns.despine(ax=ax)


#%% pie charts
df = tuningData.copy()

# only keep max tuning for each neuron
maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~df.signp, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

# create dictionary with modified alpha to separate r/o/d phases
cdict = defaultdict(lambda: np.array([1,1,1]),
                    {a:style.getColor(a[:4]) for a 
                     in ['mC2L-','mC2R-','mL2C-','mR2C-','pC2L-','pC2R-','pL2C-','pR2C-']})
cdict['pL2Cr'] = cdict['pL2C-']
cdict['pL2Co'] = np.append(cdict['pL2C-'], .45)
cdict['pL2Cd'] = np.append(cdict['pL2C-'], .7)
cdict['pR2Cr'] = cdict['pR2C-']
cdict['pR2Co'] = np.append(cdict['pR2C-'], .45)
cdict['pR2Cd'] = np.append(cdict['pR2C-'], .7)
cdict['pC2L-'] = np.append(cdict['pC2L-'], .45)

for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']

    gdata = maxdf.loc[g]   
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[cdict[a] for a in gdata.index])

    ax.set_aspect('equal')


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
axs['a2a'].set_xlabel('number of actions')


#%% TSNE
cachedDataPath = cacheFolder / "tuning_tsne.pkl"
if cachedDataPath.is_file():
    tuningTsne = pd.read_pickle(cachedDataPath)
else:
    tuningTsne = analysisTunings.getTSNEProjection(tuningData)
    tuningTsne.to_pickle(cachedDataPath)

#%%
for g,gdata in tuningTsne.groupby('genotype'):
    ax = layout.axes['tsne_'+g]['axis']
    
    ax.scatter(gdata[0], gdata[1],
               c=gdata.action.str.slice(0,4).apply(style.getColor),
               marker='.', alpha=.75, s=1.35, lw=0, clip_on=False)

    ax.set_xlim((tuningTsne[0].min(), tuningTsne[0].max()))
    ax.set_ylim((tuningTsne[1].min(), tuningTsne[1].max()))
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.axis('off')

ax = layout.axes['tsne_tuning']['axis']

ax.scatter(tuningTsne[0], tuningTsne[1],
           c=tuningTsne.action.str.slice(0,4).apply(style.getColor),
           marker='.', alpha=.75, s=3, lw=0, clip_on=False)

ax.set_xlim((tuningTsne[0].min(), tuningTsne[0].max()))
ax.set_ylim((tuningTsne[1].min(), tuningTsne[1].max()))
ax.invert_xaxis()
ax.set_aspect('equal')
ax.axis('off')


#%% similar tuning == closer spatially?
cachedDataPath = cacheFolder / "tuning_pdists.pkl"
if cachedDataPath.is_file():
    pdists = pd.read_pickle(cachedDataPath)
else:
    pdists = analysisTunings.getPDistData(endoDataPath, tuningData)
    pdists.to_pickle(cachedDataPath)
    
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
ax.legend(loc='lower right', bbox_to_anchor=(1.1, .05))
ax.set_aspect('equal')
sns.despine(ax=ax)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "tunings.svg")

