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
wAvg = (M.groupby(['shuffled','action','genotype','true','predicted'])
         .apply(analysisStaySwitchDecoding.wAvg, 'percent', 'noNeurons'))
wSem = (M.groupby(['shuffled','action','genotype','true','predicted'])
         .apply(analysisStaySwitchDecoding.bootstrap, 'percent', 'noNeurons'))

cm_df = pd.concat([wAvg, wSem], axis=1, keys=['mean','sem']).loc[False]

#%%
for (g,a), df in cm_df.groupby(['genotype','action']):
    ax = layout.axes['cm_{}_{}'.format(g,a)]['axis']
    cmap = sns.light_palette(style.getColor(a), as_cmap=True, reverse=False)
    
    cm_mean = df.unstack('predicted')['mean']
    cm_sem = df.unstack('predicted')['sem']
    cm_annot = np.apply_along_axis(lambda p: '{:.0%}\n±{:.0%}'.format(*p),
                                   2, np.stack([cm_mean, cm_sem],
                                               axis=2))
    
    sns.heatmap(cm_mean, annot=cm_annot, fmt='', cmap=cmap, square=True,
                xticklabels='', yticklabels='', cbar=False, vmin=0, vmax=1, 
                annot_kws={'fontsize':6, 'ha':'center', 'va':'center'}, ax=ax)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
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

