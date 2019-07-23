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
   
    
#%%
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
   
 
#%%
plot_action = 'mC2L'
acc_df = P.loc[P.label.str.contains('r.$|o!$')].copy() # only use win-stay, lose-switch trials
acc_df = acc_df.query('action == @plot_action & shuffled == False')

acc_groupby = acc_df.groupby(['action','duration','genotype',
                              'animal','date','noNeurons'])
n_accuracy = acc_groupby.apply(lambda s: np.mean(s.prediction == s.label))
d_accuracy = acc_groupby.apply(lambda s: np.mean(s.duration_prediction == s.label))

accuracy = pd.concat([n_accuracy, d_accuracy], keys=['activity','speed'],
                     axis=1).reset_index(['noNeurons','duration'])

#%%
ax = layout.axes['acc_v_speed']['axis']
axbp = layout.axes['acc_v_speed_bp']['axis']
ax.get_shared_x_axes().join(ax, axbp)

for gt, gdata in accuracy.groupby('genotype'):   
    means = gdata.groupby('duration')[['activity','speed']].agg(['mean','sem'])
    
    ax.fill_between(means.index, means.activity['mean']-means.activity['sem'],
                                 means.activity['mean']+means.activity['sem'],
                    lw=0, alpha=.25, color=style.getColor(gt))
    ax.plot(means.index, means.activity['mean'], 'o-', markersize=3.2,
            color=style.getColor(gt), zorder=99, alpha=.8,
            markerfacecolor='w', markeredgewidth=mpl.rcParams['lines.linewidth'])
    
    ax.fill_between(means.index, means.speed['mean']-means.speed['sem'],
                                 means.speed['mean']+means.speed['sem'],
                    lw=0, alpha=.25, color=style.getColor(gt))
    ax.plot(means.index, means.speed['mean'], 's-', markersize=2.8,
            color=sns.desaturate(style.getColor(gt),.5), zorder=99, alpha=.8,
            markerfacecolor='w', markeredgewidth=mpl.rcParams['lines.linewidth'])
    
    ax.axhline(.5, ls=':', alpha=.5, color='k', lw=mpl.rcParams['axes.linewidth'])
    
ax.set_ylim((-.05,1.05))
ax.yaxis.set_minor_locator(MultipleLocator(.25))
ax.set_yticks((0,.5,1))
ax.set_yticklabels((0,50,100))
ax.set_xlim((.27,.53))
ax.xaxis.set_minor_locator(MultipleLocator(.05))
ax.set_xticks((.3,.4,.5))
ax.set_ylabel('decoding\naccuracy (%)')
ax.set_xlabel('movement duration (s)')
sns.despine(ax=ax)

frac_wst = (acc_df.groupby(['genotype','animal','date','duration','label'])
                  .size().unstack('label').fillna(0))
frac_wst['frac'] = frac_wst[plot_action+'r.'] / \
                   (frac_wst[plot_action+'r.'] + frac_wst[plot_action+'o!'])
frac_wst = frac_wst.groupby('duration').frac.agg(['mean','sem'])

ax.errorbar(frac_wst.index, frac_wst['mean'], frac_wst['sem'],
            color='k', zorder=1000, alpha=.8, ls='--')

sns.boxplot('duration', 'label', data=acc_df,
            saturation=.85, showcaps=True,  showfliers=False,
            palette={l:style.getColor(l[-2:]) for l in acc_df.label.unique()},
            boxprops={'alpha':0.8, 'lw':0, 'zorder':-99}, width=.65, 
            whiskerprops={'c':'k','zorder':99,'clip_on':False},
            capprops={'clip_on':False}, medianprops={'c':'k','zorder':99},
            ax=axbp)
axbp.axis('off')


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecoding.svg")

