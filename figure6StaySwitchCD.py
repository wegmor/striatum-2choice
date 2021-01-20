#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:02:22 2020

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import pathlib
import figurefirst
import style
from utils import readSessions
import analysisStaySwitchDecoding
import analysisStaySwitchDecodingSupp
import subprocess

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
    
    for action in ['dL2C','pL2C','mL2C','pC2L','mC2L','dR2C','pR2C','mR2C','pC2R','mC2R']:
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
    

#%%
svgName = 'figure6StaySwitchCD.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()


#%% plot example neuron
incl_actions=['pR2C','mR2C','pC2R','pC2L']
sess = next(readSessions.findSessions(endoDataPath, animal='5308', date='190201'))
aucs = pd.read_pickle('cache/staySwitchAUC.pkl').query('animal == "5308" & date == "190201"')
aucs.set_index('action', inplace=True)
neuron = aucs.iloc[aucs.loc['mR2C','auc'].argmax()].neuron
deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)[[neuron]]
deconv.columns.name = 'neuron'
lfa = sess.labelFrameActions(reward='fullTrial', switch=True).reset_index(drop=True)
lfa['bin'] = lfa.actionProgress // (1/5)
actionMeans = deconv.groupby([lfa['label'],lfa['actionNo'],lfa['bin']]).mean()
acIndex = pd.CategoricalIndex(actionMeans.index.get_level_values(0).str.slice(0,-2), incl_actions, ordered=True, name='action')
ttIndex = pd.Index(actionMeans.index.get_level_values(0).str.slice(-2), name='trialType')
actionMeans = actionMeans.set_index([acIndex,ttIndex], append=True).reset_index('label',drop=True)
actionMeans.index = actionMeans.index.reorder_levels((1,2,3,0))
actionMeans = actionMeans.query('action in @incl_actions & trialType in ["r.","o.","o!"]')
ttMeans = actionMeans.groupby(['trialType','action','bin']).mean()


ax = layout.axes['subMean']['axis']

rst = ttMeans.loc['r.'].loc[['pR2C','mR2C','pC2R']].values
ost = ttMeans.loc['o.'].loc[['pR2C','mR2C','pC2R']].values
osw = ttMeans.loc['o!'].loc[['pR2C','mR2C','pC2L']].values
mean = np.stack([rst,ost,osw]).mean(axis=0)
x = len(mean)

ax.plot(rst, c=style.getColor('r.'))
ax.plot(ost, c=style.getColor('o.'))
ax.plot(osw,c=style.getColor('o!'), markersize=1)
ax.plot(np.arange(x*2,x*3), mean, c=style.getColor('shuffled'))
ax.plot(np.arange(x*4,x*5), rst-mean, c=style.getColor('r.'))
ax.plot(np.arange(x*4,x*5), ost-mean, c=style.getColor('o.'))
ax.plot(np.arange(x*4,x*5), osw-mean, c=style.getColor('o!'))
ax.hlines([0,0,0], [-.2*x,1.8*x,3.8*x], [1.2*x,3.2*x,5.2*x], ls=':', 
          color='k', alpha=1, lw=mpl.rcParams['axes.linewidth'])
ax.vlines(np.array([1/3,2/3,2+1/3,2+2/3,4+1/3,4+2/3])*x-.5, -.35, .85, ls=':', color='k',
          lw=mpl.rcParams['axes.linewidth'], zorder=-99)
ax.text(x*1.5, .225, '−', ha='center', va='center', fontsize=10)
ax.text(x*3.5, .225, '=', ha='center', va='center', fontsize=10)

ax.vlines(-5, -.4, -.2, color='k', lw=mpl.rcParams['axes.linewidth'], clip_on=False)
ax.hlines(-.4, -5, 0, color='k', lw=mpl.rcParams['axes.linewidth'], clip_on=False)

ax.set_title('right to center turn\ntrial type modulation\n\nOprm1+ example neuron',
             pad=6)

ax.set_ylabel('z-score')
ax.set_ylim((-.35,.85))
ax.set_xticks(())
ax.set_xlim((-.2*x,5.2*x))
ax.set_yticks((0,.5))
ax.set_yticks((0.25,), minor=True)
sns.despine(trim=True, ax=ax, bottom=True, offset=.1)
ax.axis('off')

legend_elements = [mpl.lines.Line2D([],[], color=style.getColor('r.'), alpha=1,
                                    label='win-stay'),                          
                   mpl.lines.Line2D([],[], color=style.getColor('o.'), alpha=1,
                                    label='lose-stay'),
                   mpl.lines.Line2D([], [], color=style.getColor('o!'), alpha=1,
                                    label='lose-switch'),
                  ]
ax.legend(handles=legend_elements, ncol=len(legend_elements), loc='upper center',
          bbox_to_anchor=(.5,-.15), columnspacing=1)


#%% load coding direction data (traces and means)
cdMeans = analysisStaySwitchDecodingSupp.getStSwCodingDirectionRaster(endoDataPath)
cmap = (mpl.colors.LinearSegmentedColormap
           .from_list('wStLSw', [style.getColor('o!'), (1,1,1), style.getColor('r.')], 256))
cbar = mpl.colors.LinearSegmentedColormap.from_list('test',[style.getColor(c) for c in cdMeans.columns],
                                                    len(cdMeans.columns))

cdMeansBinned, cdMeansBinnedRS = analysisStaySwitchDecodingSupp.getStSwCodingDirectionTraces(endoDataPath)
cdSEMs = (pd.concat(cdMeansBinnedRS, keys=np.arange(len(cdMeansBinnedRS)), names=['sampleNo'])
            .groupby(['genotype','trialType','action','bin']).std())


#%% plot right to center turn CD traces
rightStay = ['pR2C','mR2C','pC2R','mC2R','dR2C']
rightSwitch = ['pR2C','mR2C','pC2L','mC2L','dL2C']
rst = cdMeansBinned.loc['oprm1','r.'].loc[rightStay]
rstSem = cdSEMs.loc['oprm1','r.'].loc[rst.index]
ost = cdMeansBinned.loc['oprm1','o.'].loc[rightStay]
ostSem = cdSEMs.loc['oprm1','o.'].loc[ost.index]
osw = cdMeansBinned.loc['oprm1','o!'].loc[rightSwitch]
oswIndex = pd.CategoricalIndex(osw.index.get_level_values(0), rightSwitch, ordered=True)
osw = osw.set_index([oswIndex, osw.index.get_level_values(1)]).sort_index()
oswSem = cdSEMs.loc['oprm1','o!'].loc[osw.index]

ax = layout.axes['mR2C_cd']['axis']

ax.plot(rst['mR2C'].values, color=style.getColor('r.'))
ax.fill_between(np.arange(len(rst)), 
                rst['mR2C'].values+rstSem['mR2C'].values,
                rst['mR2C'].values-rstSem['mR2C'].values,
                color=style.getColor('r.'), alpha=.35, lw=0)
ax.plot(ost['mR2C'].values, color=style.getColor('o.'))
ax.fill_between(np.arange(len(ost)), 
                ost['mR2C'].values+ostSem['mR2C'].values,
                ost['mR2C'].values-ostSem['mR2C'].values,
                color=style.getColor('o.'), alpha=.35, lw=0)
ax.plot(osw['mR2C'].values, color=style.getColor('o!'))
ax.fill_between(np.arange(len(osw)), 
                osw['mR2C'].values+oswSem['mR2C'].values,
                osw['mR2C'].values-oswSem['mR2C'].values,
                color=style.getColor('o!'), alpha=.35, lw=0)

trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.vlines(np.arange(5,25,5)-.5, 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          transform=trans)
ax.axhline(0, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'], alpha=1,
           zorder=-99)
ax.set_yticks((-.1,0,.1))
ax.set_ylim((-.2,.2))
ax.set_xticks(())
ax.set_ylabel('right to center turn\nCD (a.u.)')
sns.despine(ax=ax, bottom=True, trim=True)


#%% plot oprm1 left win-stay traces
ax = layout.axes['leftRewardStayCD']['axis']

leftStay = ['pL2C','mL2C','pC2L','mC2L','dL2C']
rst = cdMeansBinned.loc['oprm1','r.'].loc[leftStay]
rstSem = cdSEMs.loc['oprm1','r.'].loc[rst.index]

offsets = -np.arange(len(leftStay))*.1 # square is .1 high and 5 wide
transY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
transX = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
for p, tuning in enumerate(leftStay):
    offset = offsets[p]
    ax.plot(rst[tuning].values+offset, color=style.getColor(tuning), lw=1.25,
            clip_on=False)
    ax.fill_between(np.arange(len(rst)), 
                    rst[tuning].values+rstSem[tuning].values+offset,
                    rst[tuning].values-rstSem[tuning].values+offset,
                    color=style.getColor(tuning), alpha=.35, lw=0)
    ax.fill_between(np.array([p*5,(p+1)*5])-.5, 0, 1, color=style.getColor(tuning), alpha=.075,
                    zorder=-9, lw=0, transform=transY)
    ax.fill_between(np.array([p*5,(p+1)*5])-.5, 0, -.075, color=style.getColor(tuning), alpha=1,
                    zorder=-9, lw=0, transform=transY, clip_on=False)
ax.vlines(np.arange(5,25,5)-.5, -.075, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          transform=transY)
ax.hlines(offsets, 0, 1, ls=':', color=[style.getColor(t) for t in leftStay],
          lw=mpl.rcParams['axes.linewidth'], alpha=1, transform=transX)
ax.hlines(offsets[:-1]-.05, 0, 1, color='k', lw=mpl.rcParams['axes.linewidth'],
          transform=transX, alpha=1)

ax.vlines(-8, 0.1, 0.2, color='k', lw=mpl.rcParams['axes.linewidth'], clip_on=False)
ax.hlines(0.1, -8, -3, color='k', lw=mpl.rcParams['axes.linewidth'], clip_on=False)

ax.set_ylim((-.45,.05))
ax.set_xlim((-.5,24.5))
ax.set_xticks(())
ax.set_yticks(offsets)
ax.set_yticklabels(['left outcome\n(in port) CD', 'left to center\n(turn) CD', 'center to left\n(in port) CD',
                    'center to left\n(turn) CD', 'left wait\n(in port) CD'])
[l.set_color(style.getColor(t)) for l,t in zip(ax.get_yticklabels(), leftStay)]
ax.set_xlabel('trial phase', labelpad=8)
#ax.set_ylabel('trial phase-specific\ntrial type-coding population')
ax.set_title('Oprm1+ phase-specific\ntrial type coding\n(left win-stay trial)',
             pad=10)


#%% plot matrices
cax = layout.axes['CDlegend']['axis']
vMinMax = .1

gt = 'oprm1'
ax = layout.axes[gt+'CD']['axis']
sns.heatmap(cdMeans.loc[gt].T, center=0, cmap=cmap, square=True, vmin=-vMinMax, vmax=vMinMax, ax=ax,
            cbar_ax=cax, cbar_kws={'orientation':'vertical', 'ticks':()})
ax.vlines(np.arange(0,31,5), 0, 10, color='k', clip_on=False)
ax.vlines(np.arange(0,31), 0, 10, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          clip_on=False)
ax.hlines(np.arange(1,10), 0, 30, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'],
          clip_on=False)
ax.hlines([0,5,10], 0, 30, ls='-', color='k', clip_on=False)
ax.set_yticks(())
ax.set_xticks(())
ax.set_ylabel('coding direction', labelpad=18, fontsize=8)
ax.set_xlabel('trial phase', labelpad=26, fontsize=8)
ax.set_title('Oprm1+', pad=4)#, loc='left')

cax.set_ylabel('CD (a.u.)', fontsize=6)
cax.text(0.5,1.025,str(vMinMax), ha='center', va='bottom', transform=cax.transAxes,
         fontsize=6)
cax.text(0.5,-.025,str(-vMinMax), ha='center', va='top', transform=cax.transAxes,
         fontsize=6)
# cax.tick_params(axis='x', length=0)
# cax.invert_xaxis()
cbarYAx = layout.axes['{}CD_ybar'.format(gt)]['axis']
cbarXAx = layout.axes['{}CD_xbar'.format(gt)]['axis']
cbarYAx.pcolormesh((cdMeans.columns.codes / 10)[:,np.newaxis], cmap=cbar)
cbarYAx.hlines(np.arange(1,10), 0, 1, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'])
cbarYAx.hlines([0,5,10], -2, 1, ls='-', color='k', clip_on=False)
cbarYAx.set_xlim((0,1))
cbarXAx.pcolormesh((cdMeans.loc[gt].index.get_level_values(1).codes / 10)[np.newaxis,:], cmap=cbar)
cbarXAx.vlines(np.arange(1,30), 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'])
cbarXAx.vlines([5,15,25], 1, -2, ls='-', color='k', clip_on=False)
cbarXAx.vlines([0,10,20,30], 1, -4, ls='-', color='k', clip_on=False)
cbarXAx.set_ylim((0,1))
cbarYAx.axis('off')
cbarYAx.invert_yaxis()
cbarXAx.axis('off')


for gt in ['d1','a2a']:
    ax = layout.axes[gt+'CD']['axis']
    sns.heatmap(cdMeans.loc[gt,'r.'].T, center=0, cmap=cmap, square=True, vmin=-vMinMax, vmax=vMinMax, ax=ax,
                cbar=False)
    ax.vlines(np.arange(0,11,5), 0, 10, color='k', clip_on=False)
    ax.vlines(np.arange(0,11), 0, 10, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines(np.arange(1,10), 0, 10, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines([0,5,10], 0, 10, ls='-', color='k', clip_on=False)
    ax.set_yticks(())
    ax.set_xticks(())
    ax.set_ylabel('coding direction', labelpad=18, fontsize=8)
    ax.set_xlabel('trial phase', labelpad=26, fontsize=8)
    ax.set_title({'a2a':'A2A+', 'd1':'D1+'}[gt], pad=4)

    cbarYAx = layout.axes['{}CD_ybar'.format(gt)]['axis']
    cbarXAx = layout.axes['{}CD_xbar'.format(gt)]['axis']
    cbarYAx.pcolormesh((cdMeans.columns.codes / 10)[:,np.newaxis], cmap=cbar)
    cbarYAx.hlines(np.arange(1,10), 0, 1, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'])
    cbarYAx.hlines([0,5,10], -2, 1, ls='-', color='k', clip_on=False)
    cbarYAx.set_xlim((0,1))
    cbarXAx.pcolormesh((cdMeans.loc[gt,'r.'].index.get_level_values(0).codes / 10)[np.newaxis,:], cmap=cbar)
    cbarXAx.vlines(np.arange(1,10), 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'])
    cbarXAx.vlines([5,], 1, -2, ls='-', color='k', clip_on=False)
    cbarXAx.vlines([0,10], 1, -4, ls='-', color='k', clip_on=False)
    cbarXAx.set_ylim((0,1))
    cbarYAx.axis('off')
    cbarYAx.invert_yaxis()
    cbarXAx.axis('off')


#%% plot same left win-stay traces as above as inset in oprm1 raster plot
ax = layout.axes['CDRasterInset']['axis']

transY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
transX = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
for p, tuning in enumerate(leftStay):
    offset = offsets[p]
    ax.plot(rst[tuning].values+offset, color='k', lw=.8, clip_on=False)
    ax.fill_between(np.arange(len(rst)), 
                    rst[tuning].values+rstSem[tuning].values+offset,
                    rst[tuning].values-rstSem[tuning].values+offset,
                    color='k', alpha=.35, lw=0)
ax.hlines(offsets, 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
          alpha=.5, transform=transX)
ax.set_ylim((-.45,.05))
ax.set_xlim((-.5,24.5))
ax.axis('off')


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
        ax.errorbar([.1,1,1.9][x], wAvgs[dec,shuffle], yerr=wSems[dec,shuffle],
                    color=style.getColor(a), clip_on=False,
                    marker={0:'o',1:'v',2:'s'}[x],
                    markersize={0:3.2,1:3.6,2:2.8}[x],
                    markerfacecolor='w',
                    markeredgewidth=.8)
   
    ax.plot([0.1,1,1.9], [wAvgs.loc[dec,shuffle] for dec,shuffle in decs],
            color=style.getColor(a), clip_on=False)
    
    for s, sdata in gdata.groupby(['animal','date']):
        ax.plot([0.1,1,1.9], [sdata.loc[dec,shuffle].accuracy for dec,shuffle in decs],
                color=style.getColor(a), alpha=.2,zorder=-99,
                lw=.5, clip_on=False)
    
    ax.axhline(0.5, lw=mpl.rcParams['axes.linewidth'], c='k', alpha=1, ls=':', clip_on=False)
    
    ax.set_ylim((.5,1))
    ax.set_xlim((-.35,2.35))
    ax.set_xticks(())
    ax.set_yticklabels(())
    ax.axis('off')
    if a == 'pL2C':
        ax.axis('on')
        ax.set_yticks((.5,.75,1))
        if gt == 'a2a':
            ax.set_yticklabels((50,75,100))
            ax.set_ylabel('win-stay vs. lose-switch\ndecoder accuracy (%)')
    else:
        ax.set_yticklabels(())
    
    sns.despine(ax=ax, bottom=True, offset=.5)
    
#ax = layout.axes['dec_legend']['axis']
#legend_elements = [mlines.Line2D([0], [0], marker='o', color='k', label='neural activity\n(labels shuffled)',
#                                 markerfacecolor='w', markersize=3.2,
#                                 markeredgewidth=.8),
#                   mlines.Line2D([0], [0], marker='v', color='k', label='neural activity',
#                                 markerfacecolor='w', markersize=3.6,
#                                 markeredgewidth=.8),
#                   mlines.Line2D([0], [0], marker='s', color='k', label='action duration',
#                                 markerfacecolor='w', markersize=2.8,
#                                 markeredgewidth=.8)
#                  ]
#ax.legend(handles=legend_elements, title='decoder', loc='center')
#ax.axis('off')


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
    
    ax.axhline(.5, ls=':', c='k', alpha=1, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
    ax.axvline(0, ls=':', c='k', alpha=1, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
    
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
    
    axkde.axvline(0, ls=':', c='k', alpha=1, zorder=-1,
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
    axkde.set_title({'d1':'D1+','a2a':'A2A+','oprm1':'Oprm1+'}[gt])


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
    
    ax.axhline(0, ls=':', color='k', alpha=1, lw=mpl.rcParams['axes.linewidth'],
               clip_on=False)

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
layout.write_svg(outputFolder / svgName)
subprocess.check_call(['inkscape', outputFolder / svgName,
                           '--export-pdf={}pdf'.format(outputFolder / svgName[:-3])])

