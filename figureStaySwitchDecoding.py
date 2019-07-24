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
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import pathlib
import figurefirst
import style
import analysisStaySwitchDecoding
import cmocean
plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path("data") / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()


#%%
layout = figurefirst.FigureLayout(templateFolder / "staySwitchDecoding.svg")
layout.make_mplfigures()


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
    
    for action in ['mL2C','pC2L','mC2L','mR2C','pC2R','mC2R']:
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
   
    
cachedDataPath = cacheFolder / "staySwitchAcrossDays.pkl"
if cachedDataPath.is_file():
    decodingAcrossDays = pd.read_pickle(cachedDataPath)
else:
    decodingAcrossDays = analysisStaySwitchDecoding.decodeStaySwitchAcrossDays(endoDataPath, alignmentDataPath)
    decodingAcrossDays.to_pickle(cachedDataPath)


#%%
acc = P.loc[P.label.str.contains('r.$|o!$')].copy() # only use win-stay, lose-switch trials
acc_groupby = acc.groupby(['genotype','animal','date','noNeurons','action', 'shuffled'])
acc_activity = acc_groupby.apply(lambda sess: np.mean(sess.prediction == sess.label))
acc_activity.name = 'accuracy'
acc_speed = acc_groupby.apply(lambda sess: np.mean(sess.duration_prediction == sess.label))
acc_speed.name = 'accuracy'
acc = pd.concat([acc_activity, acc_speed], keys=['activity','speed'], names=['decoder'])
acc = acc.reset_index('noNeurons')
acc = acc.reorder_levels((0,5,1,2,3,4))
   
#%%
for (gt, a), gdata in acc.groupby(['genotype','action']):
    ax = layout.axes['{}_{}'.format(gt, a)]['axis']
    
    wAvgs = (gdata.groupby(['decoder','shuffled'])
                  .apply(analysisStaySwitchDecoding.wAvg, 'accuracy', 'noNeurons'))
    wSems = (gdata.groupby(['decoder','shuffled'])
                  .apply(analysisStaySwitchDecoding.bootstrap, 'accuracy', 'noNeurons'))
    
    decs = [('activity',True), ('activity',False), ('speed',False)]
    for x, (dec,shuffle) in enumerate(decs):
#        ax.bar(x, wAvgs[dec,shuffle], yerr=wSems[dec,shuffle],
#               color=style.getColor(a), alpha=.3 if x==1 else .1, lw=0)
#        hatch = {0:'\\'*9, 2:'/'*9, 1:''}[x]
#        mpl.rcParams['hatch.color'] = style.getColor(a)
#        ax.bar(x, wAvgs[dec,shuffle], facecolor='none', alpha=.3, lw=0,
#               hatch=hatch)
        #marker = {0:'x', 1:'o', 2:'+'}[x]
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
                #lw=sdata.noNeurons[0]/500)
    
    ax.axhline(0.5, lw=mpl.rcParams['axes.linewidth'], c='k', alpha=.5, ls=':', clip_on=False)
    
    ax.set_ylim((.5,1))
    ax.set_xlim((-.35,2.35))
    #ax.yaxis.set_minor_locator(MultipleLocator(.25))
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
    
ax = layout.axes['dec_legend']['axis']
legend_elements = [mlines.Line2D([0], [0], marker='o', color='k', label='neural activity\n(labels shuffled)',
                                 markerfacecolor='w', markersize=3.2,
                                 markeredgewidth=.8),
                   mlines.Line2D([0], [0], marker='v', color='k', label='neural activity',
                                 markerfacecolor='w', markersize=3.6,
                                 markeredgewidth=.8),
                   mlines.Line2D([0], [0], marker='s', color='k', label='movement speed',
                                 markerfacecolor='w', markersize=2.8,
                                 markeredgewidth=.8)
                  ]
ax.legend(handles=legend_elements, title='decoder', loc='center')
ax.axis('off')


#%% coefficient corr matrix
order = ['mL2C','mC2L','pC2L','pC2R','mC2R','mR2C']
coefficients = (C.query('shuffled == False')
                 .set_index(["genotype","animal","date","neuron","action"])
                 .coefficient)
#%%
for genotype in ("oprm1", "d1", "a2a"):
    ax = layout.axes['{}_corr_m'.format(genotype)]['axis']
    
    #corr = coefficients.loc[genotype].unstack()[order].corr().values
    coef_grouped = (coefficients.loc[genotype].unstack()[order]
                                .groupby(['animal','date']))
    corr = coef_grouped.corr().unstack()
    weights = coef_grouped.size()
    corr = np.average(corr, axis=0, weights=weights).reshape(6,6)
    
    corr[np.triu_indices_from(corr)] = np.nan
    corr = np.ma.masked_where(np.isnan(corr), corr)

#    cm = sns.heatmap(corr, cbar=False, cmap=cmocean.cm.balance, vmin=-1, vmax=1,
#                     annot=corr, fmt='.2f', annot_kws={'fontsize':5}, ax=ax)
    cm = ax.pcolormesh(corr, cmap=cmocean.cm.balance, vmin=-1, vmax=1,
                       edgecolors='none', lw=0)
    ax.set_xlim((0,5))
    ax.set_ylim((1,6))
    ax.axis('off')

      
    pal = [style.getColor(a) for a in order]
    n = len(pal)
    for orient in ('v','h'):
        ax = layout.axes['{}_{}cbar'.format(genotype, orient)]['axis']
        ax.pcolormesh(np.arange(n).reshape(1,n) if orient=='h' 
                          else np.arange(n).reshape(n,1),
                      cmap=mpl.colors.ListedColormap(list(pal)),
                      alpha=.8, edgecolors='none', lw=0)
        if orient=='h':
            ax.set_xlim((0,5))
        else:
            ax.set_ylim((1,6))
        ax.axis('off')

cax = layout.axes['corr_colorbar']['axis']
cb = plt.colorbar(cm, cax=cax, orientation='horizontal',
                  ticks=(-1,1))
cax.tick_params(axis='x', which='both',length=0)
cb.outline.set_visible(False)


#%%
#
#    axfv = layout.axes['f8_{}'.format(p+1)]['axis']
#    fv = fancyViz.SchematicIntensityPlot(s, splitReturns=False,
#                                         linewidth=mpl.rcParams['axes.linewidth'],
#                                         smoothing=7)
#    img = fv.draw(trace, ax=axfv)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecoding.svg")

