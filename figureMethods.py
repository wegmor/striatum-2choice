#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:26:25 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import figurefirst
import style
import analysisMethods


#%%
style.set_context()
endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"

#%% TODO: only works if in top folder of repository! __file__ doesn't work inside Spyder!
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "methods.svg")
layout.make_mplfigures()


#%% peri-outcome swap choice behavior plot
win_size = (10,30)
sdf = analysisMethods.getPeriSwapChoices(endoDataPath, win_size)
#%%
ax = layout.axes['out_swap']['axis']

ax.axhline(.5, color='k', ls=':', zorder=-99, alpha=.5, lw=mpl.rcParams['axes.linewidth'])
ax.axvline(-.5, color='k', ls=':', zorder=-99, alpha=.5, lw=mpl.rcParams['axes.linewidth'])

for animal, adata in sdf.groupby('animal'):
    ax.plot(adata.loc[animal,'R'], color=style.getColor('a2a'), alpha=.2)
    ax.plot(adata.loc[animal,'L'], color=style.getColor('d1'), alpha=.2)
 
df = sdf.groupby(['rewardP','swapDist']).leftIn.agg(['mean','sem'])
df = df.reset_index(['swapDist'])

ax.fill_between(df.loc['R','swapDist'], 
                df.loc['R','mean']-df.loc['R','sem'],
                df.loc['R','mean']+df.loc['R','sem'], 
                color=style.getColor('a2a'), alpha=.6, lw=0, zorder=1)
ax.plot(df.loc['R','swapDist'], df.loc['R','mean'], color=style.getColor('a2a'),
        label=r'right $\rightarrow$ left', zorder=1)
ax.fill_between(df.loc['L','swapDist'], 
                df.loc['L','mean']-df.loc['L','sem'],
                df.loc['L','mean']+df.loc['L','sem'], 
                color=style.getColor('d1'), alpha=.6, lw=0)
ax.plot(df.loc['L','swapDist'], df.loc['L','mean'], color=style.getColor('d1'),
        label=r'left $\rightarrow$ right')

ax.legend(bbox_to_anchor=(1,1), loc='upper right')

#ax.set_ylim((0,1))
ax.set_yticks((0,.25,.5,.75,1))
ax.set_yticklabels((0,25,50,75,100))
ax.set_ylabel('% left port entries')
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.set_xticks([-10]+list(range(9,40,10)))
ax.set_xticklabels([-10]+list(range(10,41,10)))
ax.set_xlim((-win_size[0], win_size[1]-1))
ax.set_xlabel('choice port entry\n(relative to outcome swap)')
sns.despine(ax=ax)


#%% behavior stats boxplots
cdf = analysisMethods.getChoiceData(endoDataPath)
#%%
ax = layout.axes['beh_stats']['axis']

bstats = cdf.groupby('animal')[['leftEx','correct','triggered','reward','switch']].sum() / \
         cdf.groupby('animal')[['leftEx','correct','triggered','reward','switch']].count()

palette = {'leftEx':style.getColor('pL'), 'correct':style.getColor('correct'),
           'reward':style.getColor('stay'), 'switch':style.getColor('switch'),
           'triggered':style.getColor('error')}

sns.swarmplot(data=bstats,
              size=3.8, palette=palette, marker='.', ax=ax)
sns.boxplot(data=bstats,
            saturation=.85, showcaps=False,  showfliers=False,
            boxprops={'alpha':0.35, 'lw':0, 'zorder':-99}, width=.55, palette=palette,
            whiskerprops={'c':'k','zorder':99}, medianprops={'c':'k','zorder':99},
            ax=ax)

ax.axhline(.5, ls=':', alpha=.5, color='k', zorder=-99)
ax.set_ylim((0,1))
ax.set_yticks((0,.25,.5,.75,1))
ax.set_yticklabels((0,25,50,75,100))
ax.set_ylabel('% side port entries')
ax.set_xticks(np.arange(5))
ax.set_xticklabels(['left port','correct port','delay error','rewarded','switch'],
                    rotation=55, ha='right')
ax.set_xlim((-.8,4.8))
sns.despine(ax=ax)


#%%
ax1 = layout.axes['rel_reward_bp']['axis']
ax2 = layout.axes['rel_reward']['axis']

sstats = (cdf.groupby(['animal','toSwitch']).reward.mean())

astats = sstats.unstack('animal').loc[-10:0]
gstats = sstats.groupby('toSwitch').agg(['mean','sem']).loc[-10:0]

sns.boxplot(data=astats.loc[-3:0].T, ax=ax1,
            saturation=.85, showcaps=False,  showfliers=False,
            boxprops={'alpha':0.3, 'lw':0, 'zorder':-99, 'clip_on':False}, 
            width=.3, color=style.getColor('stay'),
            whiskerprops={'c':'k','zorder':99, 'clip_on':False},
            medianprops={'c':'k','zorder':99, 'clip_on':False})
ax1.plot([0,1,2,3], astats.loc[-3:0], '-', color=style.getColor('stay'), alpha=.6,
         clip_on=False, lw=.35)
#ax1.vlines(3, .05, .3, linestyle=':', color=style.getColor('switch'), zorder=-99,
#           alpha=1)

ax1.set_xlabel('')
ax1.set_ylim((.05,.3))
ax1.set_xticks(())
ax1.set_yticks((.1,.2,.3))
ax1.set_yticklabels((10,20,30))
ax1.set_zorder(99)
sns.despine(ax=ax1, bottom=True, left=False, right=True, trim=True)

e=ax2.errorbar(gstats.index, gstats['mean'], gstats['sem'], clip_on=False,
               color=style.getColor('stay'))
e[0].set_clip_on(False)
[b.set_clip_on(False) for b in e[2]]
ax2.axhline(bstats.switch.mean(), alpha=.5, color=style.getColor('stay'))
ax2.fill_between([-11,1], bstats.switch.mean()-bstats.switch.sem(),
                          bstats.switch.mean()+bstats.switch.sem(),
                          alpha=.2, color=style.getColor('stay'), lw=0)
ax2.vlines(0, .05, .49, linestyle=':', color=style.getColor('switch'), zorder=-99,
           alpha=1, clip_on=False)
ax2.vlines(-3.5, .05, .28, linestyle=':', color='k', zorder=100)

ax2.set_xticks((-10,-5,0))
ax2.set_xlim((-11,1))
ax2.set_xlabel('side port entry\n(relative to switch)')
ax2.set_yticks(np.arange(0.1,.35,.1))
ax2.set_yticklabels(np.arange(10,35,10))
ax2.set_ylim((.05,.3))
ax2.set_ylabel('% rewarded')
sns.despine(ax=ax2, trim=False)

analysisMethods.align_xaxis(ax2, 0, ax1, 3)


#%%
ax1 = layout.axes['rel_switch_bp']['axis']
ax2 = layout.axes['rel_switch']['axis']

sstats = (cdf.groupby(['animal','sinceReward']).switch.mean())

astats = sstats.unstack('animal').loc[0:10]
gstats = sstats.groupby('sinceReward').agg(['mean','sem']).loc[0:10]

sns.boxplot(data=astats.loc[0:3].T, ax=ax1,
            saturation=.85, showcaps=False,  showfliers=False,
            boxprops={'alpha':0.3, 'lw':0, 'zorder':-99, 'clip_on':False}, 
            width=.3, color=style.getColor('switch'),
            whiskerprops={'c':'k','zorder':99, 'clip_on':False},
            medianprops={'c':'k','zorder':99, 'clip_on':False})
ax1.plot(astats.loc[0:3], '-', color=style.getColor('switch'), alpha=.6,
         clip_on=False, lw=.35)
#ax1.vlines(0, .05, .3, linestyle=':', color=style.getColor('stay'), zorder=-99,
#           alpha=1)

ax1.set_xlabel('')
ax1.set_ylim((.05,.3))
ax1.set_xticks(())
ax1.set_yticks((.1,.2,.3))
ax1.set_yticklabels((10,20,30))
ax1.set_zorder(99)
sns.despine(ax=ax1, bottom=True, left=True, right=False, trim=True)
ax1.yaxis.tick_right()

ax2.errorbar(gstats.index, gstats['mean'], gstats['sem'], clip_on=False,
             color=style.getColor('switch'))
ax2.axhline(bstats.switch.mean(), alpha=.5, color=style.getColor('switch'))
ax2.fill_between([-1,11], bstats.switch.mean()-bstats.switch.sem(),
                          bstats.switch.mean()+bstats.switch.sem(),
                          alpha=.2, color=style.getColor('switch'), lw=0)
ax2.vlines(0, .05, .49, linestyle=':', color=style.getColor('stay'), zorder=-99,
           alpha=1, clip_on=False)
ax2.vlines(3.5, .05, .28, linestyle=':', color='k', zorder=100)

ax2.set_xticks((0,5,10))
ax2.set_xlim((-1,11))
ax2.set_xlabel('side port entry\n(relative to reward)')
ax2.set_yticks(np.arange(0.1,.35,.1))
ax2.set_yticklabels(np.arange(10,35,10))
ax2.set_ylim((.05,.3))
ax2.set_ylabel('% switches')
sns.despine(ax=ax2, trim=False)

analysisMethods.align_xaxis(ax2, 0, ax1, 0)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "methods.svg")