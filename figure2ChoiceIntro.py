#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:40:58 2020

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
import analysis2ChoiceIntro


#%%
endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "2ChoiceIntro.svg")
layout.make_mplfigures()


#%% peri-outcome swap choice behavior plot
win_size = (10,30)
sdf = analysis2ChoiceIntro.getPeriSwapChoices(endoDataPath, win_size)
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
cdf = analysis2ChoiceIntro.getChoiceData(endoDataPath)
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

analysis2ChoiceIntro.align_xaxis(ax2, 0, ax1, 3)


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

analysis2ChoiceIntro.align_xaxis(ax2, 0, ax1, 0)


#%%
trialType = 'r.'
df = mean_rs.query('trialType == @trialType').copy().reset_index('trialType', drop=True)
df = df.reset_index('action')
if trialType.endswith('.'):
    df['action'] = pd.Categorical(df.action, categories=['pL2C','mL2C','pC2L','mC2L','dL2C',
                                                         'pR2C','mR2C','pC2R','mC2R','dR2C'],
                                  ordered=True)
else:
    df['action'] = pd.Categorical(df.action, categories=['pL2C','mL2C','pC2R','mC2R','dR2C',
                                                         'pR2C','mR2C','pC2L','mC2L','dL2C'],
                                  ordered=True)
df = df.set_index('action', append=True).unstack('action')
df.columns = df.columns.reorder_levels([1,0])
df = df.sort_index(axis=1).sort_index(axis=0)

df /= mean_rs.unstack(['action','trialType']).abs().max(axis=1).values[:,np.newaxis]
#df /= df.abs().max(axis=1).values[:,np.newaxis]

#%%
idx = pd.DataFrame({'idxmin':np.argmin(df.values, axis=1), 
                    'idxmax':np.argmax(df.values, axis=1), 
                    'posPeak':(df.max(axis=1) > df.min(axis=1).abs()).values},
                   index=df.index)

def getSorting(row, no_cols):
    return (row['idxmax'] - no_cols) if row.posPeak else (np.abs(row['idxmin'] - no_cols))

idx['sorting'] = idx.apply(getSorting, axis=1, no_cols=df.shape[1])
idx['right'] = idx.sorting.abs() <= (df.shape[1] // 2)
idx = idx.sort_values(['right','sorting'])

#%%
posCount = pd.DataFrame(idx.loc[idx.posPeak].groupby(['genotype','idxmax']).size(),
                        columns=['count']).reset_index()
posCount['bin'] = posCount['idxmax'] // (df.shape[1] // 10 // 2)
posCount = posCount.groupby(['genotype','bin'])[['count']].sum()
posCount['count'] /= idx.groupby('genotype').size()
posCount = posCount.unstack('genotype')['count']
posCount = posCount.reindex(np.arange(df.shape[1] // (df.shape[1] // 10 // 2)))
posCount['action'] = np.stack([df.columns.levels[0], df.columns.levels[0]], axis=1).flatten()
posCount.reset_index(inplace=True)
posCount['bin'] = posCount['bin'] % 2
posCount.set_index(['action','bin'], inplace=True)
posCount = posCount[['d1','a2a','oprm1']]

#%%
df = df.loc[idx.index]

barcol = [style.getColor(gt) for gt in posCount.columns]
no_bins = df['pL2C'].shape[1]
for action in df.columns.levels[0]:
    barAx = layout.axes['bar_{}'.format(action)]['axis']
    barAx.bar(np.arange(-3,0), posCount.loc[action,0].values, color=barcol)
    barAx.bar(np.arange(1,4), posCount.loc[action,1].values, color=barcol)
    barAx.axvline(0, c='k', ls=':')
    barAx.set_ylim((0,.2))
    barAx.axis('off')
    if action == 'dR2C':
        barAx.axis('on')
        barAx.set_xticks(())
        sns.despine(bottom=True, left=True, right=False, ax=barAx,
                    offset=1)
        barAx.set_yticks((0,.1,.2))
        barAx.set_yticklabels((0,10,20))
        barAx.set_yticks((.05, .15), minor=True)
        barAx.set_ylabel('% neurons')
        barAx.yaxis.set_label_position('right')
        barAx.yaxis.set_ticks_position('right')
    
    rasterAx = layout.axes['raster_{}'.format(action)]['axis']
    img = rasterAx.imshow(df.loc[:,action].values, aspect='auto',
                          vmin=-1, vmax=1, cmap='RdBu_r')
    rasterAx.axvline(no_bins // 2 - .5, c='k', ls=':')
    rasterAx.axis('off')
    if action == 'pL2C':
        rasterAx.axis('on')
        sns.despine(left=True, bottom=True, ax=rasterAx)
        rasterAx.set_xticks(())
        rasterAx.set_yticks((0, len(df)-1))
        rasterAx.set_yticklabels([1,len(df)])
        rasterAx.set_ylabel('neuron', labelpad=-15)
        rasterAx.tick_params(axis='y', length=0, pad=2)
    if action in ['pC2L','pC2R']:
        rasterAx.axis('on')
        sns.despine(left=True, ax=rasterAx, offset=1)
        rasterAx.set_yticks(())
        rasterAx.set_xticks((-.5, no_bins // 2 - .5, no_bins - .5))
        rasterAx.set_xticklabels((-no_bins // 2 * 50, 0, no_bins // 2 * 50))
    
cax = layout.axes['raster_colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='vertical')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(.38, 1.032, '1', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.25, -.05, '-1', ha='center', va='top', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(3.45, .5, 'peak-normalized\naverage', ha='center', va='center', fontdict={'fontsize':7},
         rotation=90, transform=cax.transAxes)
        
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "2ChoiceIntro.svg")