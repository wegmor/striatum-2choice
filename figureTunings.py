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
from matplotlib.ticker import MultipleLocator
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
cachedDataPath = cacheFolder / "actionTunings.pkl.bak"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisTunings.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)

tuningData['signp'] = tuningData['pct'] > .99
tuningData['signn'] = tuningData['pct'] < .01
tuningData['tuning'] = (tuningData['mean']-tuningData['s_mean'])/tuningData['s_std']

#%%
ex_session = ('oprm1','5308','190131')
s = next(readSessions.findSessions(endoDataPath, genotype=ex_session[0],
                                   animal=ex_session[1], date=ex_session[2],
                                   task='2choice'))
traces = s.readDeconvolvedTraces(zScore=True)
lfa = s.labelFrameActions(reward='sidePorts').set_index(traces.index)
actions = ['pC2L','mC2R','pL2Cr']
tunings = tuningData.query("genotype == @ex_session[0] & animal == @ex_session[1] & "+
                           "date == @ex_session[2] & action in @actions").copy()

sel_traces = {}
for p,(a,t) in enumerate(tunings.groupby('action')):
    max_neuron = t.loc[t.tuning.idxmax(),'neuron']
    trace = traces[max_neuron]
    sel_traces[a] = trace
    
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


#%%
ax1 = layout.axes['tuning_hist1']['axis']

hdata = tuningData.query('genotype == "oprm1" & action == "mC2L"').copy()
ax1.hist(hdata['tuning'], bins=np.arange(-20,40,1), lw=0, color='gray', alpha=.6,
         histtype='stepfilled')
ax1.hist(hdata.loc[hdata.signp,'tuning'], np.arange(-20,40,1), lw=0,
         histtype='stepfilled', color=style.getColor('mC2L'))
ax1.text(30,45,'significant\ntuning',ha='right',va='bottom',fontdict={'fontsize':7},
         color=style.getColor('mC2L'))
ax1.text(7.5,400,'center to left\nturn',ha='center',va='center',fontdict={'fontsize':7})

#ax2 = layout.axes['tuning_hist2']['axis']
#ax2.get_shared_x_axes().join(ax1, ax2)
#ax2.get_shared_y_axes().join(ax1, ax2)
#
#hdata = tuningData.query('genotype == "oprm1" & action == "mC2R"').copy()
#ax2.hist(hdata['tuning'], bins=np.arange(-20,40,1), lw=0, color='gray', alpha=.6,
#         histtype='stepfilled')
#ax2.hist(hdata.loc[hdata.signp,'tuning'], np.arange(-20,40,1), lw=0,
#         histtype='stepfilled', color=style.getColor('mC2R'))
#ax2.text(7.5,500,'center to right\nturn',ha='center',va='center',fontdict={'fontsize':6})

ax1.set_yticks((0,200,400))
#ax2.set_yticks((0,250,500))
ax1.set_xticks((-15,0,15,30))
#ax2.set_xticks((-15,0,15,30))
ax1.set_xlim((-15,30))
ax1.set_ylim((0,400))
#ax1.set_xticklabels(())
ax1.set_xlabel('tuning score')

sns.despine(ax=ax1)
#sns.despine(ax=ax2)


#%% pie charts
df = tuningData.copy()

# only keep max tuning for each neuron
maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~df.signp, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

# create dictionary with modified alpha to separate r/o/d phases
cdict = defaultdict(lambda: np.array([1,1,1]),
                    {a:style.getColor(a) for a 
                     in ['mC2L','mC2R','mL2C','mR2C','pC2L','pC2R','pL2C','pR2C']})
cdict['pL2Cr'] = cdict['pL2C']
cdict['pL2Co'] = np.append(cdict['pL2C'], .45)
cdict['pL2Cd'] = np.append(cdict['pL2C'], .7)
cdict['pR2Cr'] = cdict['pR2C']
cdict['pR2Co'] = np.append(cdict['pR2C'], .45)
cdict['pR2Cd'] = np.append(cdict['pR2C'], .7)
cdict['pC2L'] = np.append(cdict['pC2L'], .45)

for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']

    gdata = maxdf.loc[g]   
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[cdict[a] for a in gdata.index])

    ax.set_aspect('equal')
    #ax.set_xlabel(g)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "tunings.svg")


#%% tuning charts
#df = tuningData.copy()
#
#df['signp'] = df['pct'] > .99
#df['signn'] = df['pct'] < .01
#df['sign'] = df.signp.astype('int') - df.signn.astype('int')
#
#sign_count = (df.groupby(['genotype','animal','date','action'])
#                .agg({'signp':'sum','signn':'sum'}))
#total_count = (df.groupby(['genotype','animal','date','action'])
#                 [['signp','signn']].count())
#sign_pct = sign_count / total_count
#sign_pct['noNeurons'] = total_count.signp
#
##%%
#actions = ['pC2L-','mC2L-','pL2Cd','pL2Co','pL2Cr','mL2C-',
#           'pC2R-','mC2R-','pR2Cd','pR2Co','pR2Cr','mR2C-']
#
## v x coords for actions
#a2x = dict(zip(actions, np.arange(12)))
#sign_pct['x'] = sign_pct.reset_index().action.replace(a2x).values
## v color for actions
#sign_pct['color'] = sign_pct.reset_index().action.str.slice(0,4).apply(style.getColor).values
#
#fig, axs = plt.subplots(1, 3, figsize=(2, 1), sharey=True, sharex=True,
#                        gridspec_kw={'wspace': .1, 'hspace': .2})
#
#for g, gdata in sign_pct.groupby('genotype'):
#    ax = axs[{'d1':0, 'oprm1':2, 'a2a': 1}[g]]
#    ax.scatter(analysisTunings.jitter(gdata.x, .12), gdata.signp, s=gdata.noNeurons/5,
#               edgecolor=gdata.color, facecolor='none')
#    
#    avg = gdata.groupby('x').apply(analysisTunings.wAvg, 'signp', 'noNeurons')
#    sem = gdata.groupby('x').apply(analysisTunings.bootstrap, 'signp', 'noNeurons')
#    ax.errorbar(avg.index, avg, sem, fmt='.-', c='k')
#    
#    for a, adata in gdata.groupby('animal'):
#        avg = adata.groupby('x').apply(analysisTunings.wAvg, 'signp', 'noNeurons')
#        lw = adata.groupby('date').noNeurons.first().sum() / 400
#        ax.plot(avg, c='k', alpha=.35, lw=lw)
#    
#    ax.set_xticks(np.arange(12))
#    #ax.set_xticklabels(actions, rotation=90)
#    ax.set_title(g)
#    sns.despine()
#
#axs[0].set_ylabel('pos. tuned (%)')
#axs[0].set_yticks((0,.4,.8))
#axs[0].yaxis.set_minor_locator(MultipleLocator(.2))
#axs[0].set_yticklabels((0,40,80))
#axs[0].set_ylim((0,.8))
