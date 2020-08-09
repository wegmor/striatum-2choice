#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:49:30 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib.ticker import MultipleLocator, FixedLocator
import pathlib
import figurefirst
import style
import analysisTunings
import analysisRewardSupp
import analysisMethods
plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "rewardSupp.svg")
layout.make_mplfigures()


#%% load tuning data
tuningData = analysisTunings.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005

#%% Panel A
cdf = analysisMethods.getChoiceData(endoDataPath)
##%%
ax1 = layout.axes['rel_switch_bp']['axis']
ax2 = layout.axes['rel_switch']['axis']

sstats1 = (cdf.loc[cdf.rewardx == 1].groupby(['animal','sinceReward']).switch.mean())
sstats2 = (cdf.loc[cdf.rewardx == 2].groupby(['animal','sinceReward']).switch.mean())

astats = pd.concat([sstats1.loc[:,:3], sstats2.loc[:,:3]], axis=0,
                   keys=[1,2], names=['rewardx'])
gstats1 = sstats1.groupby('sinceReward').agg(['mean','sem']).loc[0:10]
gstats2 = sstats2.groupby('sinceReward').agg(['mean','sem']).loc[0:10]

sns.boxplot(x='sinceReward', y='switch', hue='rewardx', data=astats.reset_index(),
            saturation=.85, showcaps=False,  showfliers=False,
            boxprops={'alpha':0.4, 'lw':0, 'zorder':-99, 'clip_on':False}, 
            width=1, palette={1:style.getColor('r.'), 2:style.getColor('double')},
            whiskerprops={'c':'k','zorder':99, 'clip_on':False},
            medianprops={'c':'k','zorder':99, 'clip_on':False}, ax=ax1)
ax1.legend_.remove()
for x, data in astats.unstack('rewardx').groupby('sinceReward'):
    ax1.plot(np.array([[x-.25,x+.25]]*len(data)).T, data.values.T, c='k',
             alpha=1, lw=.5, zorder=99, clip_on=False)

ax1.set_xlabel('')
ax1.set_ylim((.05,.3))
ax1.set_xlim((-.6, 3.6))
ax1.set_xticks(())
ax1.set_yticks((.1,.2,.3))
ax1.set_yticklabels((10,20,30))
ax1.set_zorder(99)
ax1.set_ylabel('')
sns.despine(ax=ax1, bottom=True, left=True, right=False, trim=True)
ax1.yaxis.tick_right()

ax2.errorbar(gstats1.index, gstats1['mean'], gstats1['sem'], clip_on=False,
             color=style.getColor('r.'), capsize=1, alpha=1, label=' '*6)
ax2.errorbar(gstats2.index, gstats2['mean'], gstats2['sem'], clip_on=False,
             color=style.getColor('double'), capsize=1, alpha=1, label=' '*6)
ax2.vlines(0, .05, .53, linestyle=':', color=style.getColor('r.'), zorder=-99,
           alpha=1, clip_on=False)
ax2.vlines(3.5, .05, .28, linestyle=':', color='k', zorder=100)

ax2.legend(bbox_to_anchor=(1,.2), loc='center right', labelspacing=.4)

ax2.set_xticks((0,5,10))
ax2.set_xlim((-1,11))
ax2.set_xlabel('side port entry\n(relative to reward)')
ax2.set_yticks(np.arange(0.1,.35,.1))
ax2.set_yticklabels(np.arange(10,35,10))
ax2.set_ylim((.05,.3))
ax2.set_ylabel('% switches')
sns.despine(ax=ax2, trim=False)

analysisMethods.align_xaxis(ax2, 0, ax1, 0)


#%% Panel B
lt_avg_trial, rt_avg_trial, lt_avg_frame, rt_avg_frame = \
        analysisRewardSupp.getRewardRespData(endoDataPath, tuningData)
##%% 
for genotype in ['d1','a2a','oprm1']:
    resp_lt = (lt_avg_trial.query('genotype == @genotype')
                           .set_index(['side','reward','animal','date','neuron'])[['avg']]
                           .unstack(['animal','date','neuron']))
    resp_rt = (rt_avg_trial.query('genotype == @genotype')
                           .set_index(['side','reward','animal','date','neuron'])[['avg']]
                           .unstack(['animal','date','neuron']))
    
    for (side, df) in zip(('L','R'), (resp_lt, resp_rt)):
        ax = layout.axes['{}_pointplot_{}'.format(genotype, {'L':'lt','R':'rt'}[side])]

        sns.stripplot(data=(df.loc[side].loc[[0,2]] - df.loc[side,1]).T,
                      ax=ax, color=style.getColor('p'+side),
                      size=1.5, alpha=.5, jitter=.32, linewidth=0, zorder=-1,
                      clip_on=False)
        means = (df.loc[side].loc[[0,1,2]] - df.loc[side,1]).T.mean()
        sems = (df.loc[side].loc[[0,1,2]] - df.loc[side,1]).T.sem()
        ax.errorbar([0,.5,1], means, yerr=sems,
                    fmt='ko-', markersize=3.2, markeredgewidth=0)
        
        ax.axhline(0, ls='--', zorder=-5, c='k', alpha=.35, lw=mpl.rcParams['axes.linewidth'])
        ax.axvline(.5, ls='--', zorder=-5, c='k', alpha=.35, lw=mpl.rcParams['axes.linewidth'])
        
        ax.text(.975, .99, 'n={}'.format(df.shape[1]), fontsize=6, ha='right',
                va='top', transform=ax.transAxes)

        ax.set_xticks((0,.5,1))
        ax.set_xticklabels(())
        ax.set_xlabel('')
        if genotype == 'd1' and side == 'L':
            ax.set_ylabel('Δsd')
        ax.set_yticks((-.5,0,.5))
        if not (genotype == 'd1' and side == 'L'):
            ax.set_yticklabels(())
        ax.set_yticks((-.25,.25), minor=True)
        ax.set_ylim((-.8,.8))
    
        sns.despine(ax=ax, trim=True)


#%%
for gt in ['d1','a2a','oprm1']:
    lt_df = (lt_avg_frame.query('genotype == @gt')
                         .set_index(['side','reward','animal','date','neuron','actionFrame'])
                         .sort_index().unstack('actionFrame')['avg'])
    rt_df = (rt_avg_frame.query('genotype == @gt')
                         .set_index(['side','reward','animal','date','neuron','actionFrame'])
                         .sort_index().unstack('actionFrame')['avg'])
    
    for (side, df) in zip(('L','R'), (lt_df, rt_df)):
        ax = layout.axes['{}_trace_{}'.format(gt, {'L':'lt','R':'rt'}[side])]
        
        df = df.query('side == @side').groupby('reward').agg(['mean','sem']).loc[:,:36].copy()
        df.columns = df.columns.reorder_levels((1,0))
        
        ax.axvline(-.5, ls='--', c='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
        ax.axvline(6.5, ls='--', c='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
        ax.axhline(0, ls='--', c='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
        
        ax.fill_between(np.arange(26),
                        df.loc[0,'mean'][:26]+df.loc[0,'sem'][:26],
                        df.loc[0,'mean'][:26]-df.loc[0,'sem'][:26],
                        alpha=.25, lw=0, clip_on=False, zorder=-3,
                        color=style.getColor('o!'))
        ax.plot(df.loc[0,'mean'][:26], color=style.getColor('o!'), alpha=.8,
                zorder=1, label=' '*6)
        ax.fill_between(np.arange(37),
                        df.loc[1,'mean']+df.loc[1,'sem'],
                        df.loc[1,'mean']-df.loc[1,'sem'],
                        alpha=.25, lw=0, clip_on=False, zorder=-1,
                        color=style.getColor('r.'))
        ax.plot(df.loc[1,'mean'], color=style.getColor('r.'), alpha=.8, 
                zorder=3, label=' '*6)
        ax.fill_between(np.arange(37),
                        df.loc[2,'mean']+df.loc[2,'sem'],
                        df.loc[2,'mean']-df.loc[2,'sem'],
                        alpha=.25, lw=0, clip_on=False, zorder=-2,
                        color=style.getColor('double'))
        ax.plot(df.loc[2,'mean'], color=style.getColor('double'), alpha=.8,
                zorder=2, label=' '*6)
        
        ax.set_ylim((-.25,.5))
        ax.set_xlim((-3.5,36.5))
        ax.set_yticks((-.25,0,.25,.5))
        ax.set_xticks((-.5,6.5,16.5,26.5,36.5))
        ax.set_xticklabels(('','',.5,1.,1.5), rotation=0)
        if not (gt == 'd1' and side == 'L'):
            ax.set_yticklabels(())
        else:
            ax.set_ylabel('sd')
            ax.legend(bbox_to_anchor=(1,.2), loc='center right')
        
        sns.despine(ax=ax, trim=False)

    
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "rewardSupp.svg")
