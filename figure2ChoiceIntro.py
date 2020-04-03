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
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import ArrowStyle
import pathlib
import figurefirst
import style
import analysis2ChoiceIntro
import subprocess
plt.ioff()
style.set_context()


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
svgName = 'figure2ChoiceIntro.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName, dpi=600)
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

meta = pd.read_hdf(endoDataPath, "/meta").query("task == '2choice'")
print("Panels B-E:")
print("animals:", meta.animal.nunique())
print("sessions:", len(meta))
print("\tof which recorded:", len(meta.query("camera == 'on'")))


#%% rasterplot
mean_rs = analysis2ChoiceIntro.getRasterAverages(endoDataPath, smooth=True)
mean_rs.set_index(['genotype','animal','date','neuron','action','trialType'], inplace=True)
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
                          vmin=-1, vmax=1, cmap='RdBu_r', interpolation='nearest')
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

axl = layout.axes['bar_legend']['axis']
axl.axis('off')
patches = [mpatches.Patch(color=style.getColor(g),
                          label={'oprm1':'Oprm1','a2a':'A2A','d1':'D1'}[g], alpha=1) 
               for g in ['d1','a2a','oprm1']]
axl.legend(handles=patches, ncol=3, mode='expand', frameon=True, framealpha=1,
           edgecolor='w', facecolor='w', borderpad=.1)

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
# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
    
def plot3D(train_fits, test_fits, azimuth, angle, lims=None, bins=4, order=['r.','o.','o!'],
           a1_offsets=[0,0,0], a2_offsets=[0,0,0], a3_offsets=[0,0,0],
           t1_offsets=[0,0,0], t2_offsets=[0,0,0], t3_offsets=[0,0,0],
           t4_offsets=[0,0,0], t5_offsets=[0,0,0], ax=None):
    def _arrowD(a, b):
        d = b - a
        dnorm = d / np.linalg.norm(d)
        return dnorm
    
    if not ax:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azimuth, angle)
     
    if lims: 
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_zlim(lims)
    
    # thin lines: bootstrapped trials
    for (tt,it,tn), df in test_fits.groupby(['trialType','iteration','trialNo']):
        ax.plot(df[0], df[1], df[2], c=style.getColor(tt[-2:]),
                alpha=.07, lw=.35, zorder={'r.':3, 'o.':2, 'o!':1}[tt[-2:]])

    # average line of PCA'ed trials
    meandf = train_fits.groupby(['trialType','bin']).mean().reset_index()
    for tt, df in meandf.groupby('trialType'):
        ax.plot(df[0], df[1], df[2], c=style.getColor(tt[-2:]),
                alpha=.6, lw=.8, zorder={'r.':9, 'o.':6, 'o!':3}[tt[-2:]])
        ax.scatter(df[0], df[1], df[2], color=style.getColor(tt[-2:]),
                   alpha=.6, s=4, lw=0, zorder={'r.':8, 'o.':5, 'o!':2}[tt[-2:]])
        df = df.loc[df.bin % bins == 0]
        ax.scatter(df[0], df[1], df[2], color=style.getColor(tt[-2:]),
                   alpha=.8, s=10, lw=0, zorder={'r.':7, 'o.':4, 'o!':1}[tt[-2:]])
    
    # arrows
    ab = meandf.loc[meandf.trialType.str.endswith('r.') & meandf.bin.isin([2,3]), [0,1,2]].values
    d = _arrowD(*ab)
    ac = np.stack([ab[0], ab[0]+d], axis=1) + np.stack([a1_offsets]*2, axis=1)
    a1 = Arrow3D(*ac, lw=1, arrowstyle=ArrowStyle('-|>', head_length=3, head_width=1.5),
                 color='k', zorder=999)
    ax.add_artist(a1)
    
    ab = meandf.loc[meandf.trialType.str.endswith('r.') & meandf.bin.isin([5,6]), [0,1,2]].values
    d = _arrowD(*ab)
    ac = np.stack([ab[0], ab[0]+d], axis=1) + np.stack([a2_offsets]*2, axis=1)
    a2 = Arrow3D(*ac, lw=1, arrowstyle=ArrowStyle('-|>', head_length=3, head_width=1.5),
                 color='k', zorder=999)
    ax.add_artist(a2)
    
    ab = meandf.loc[meandf.trialType.str.endswith('r.') & meandf.bin.isin([9,11]), [0,1,2]].values
    d = _arrowD(*ab)
    ac = np.stack([ab[0], ab[0]+d], axis=1) + np.stack([a3_offsets]*2, axis=1)
    a3 = Arrow3D(*ac, lw=1, arrowstyle=ArrowStyle('-|>', head_length=3, head_width=1.5),
                 color='k', zorder=999)
    ax.add_artist(a3)
    
    # text annotations indicating phase
    phase_nos = [u"\u2460",u"\u2461",u"\u2462",u"\u2463",u"\u2464"]
    coords = (meandf.loc[meandf.trialType.str.endswith('r.') & meandf.bin.isin(np.arange(5)*bins),
                         [0,1,2]].values)
    coords += np.stack([t1_offsets,t2_offsets,t3_offsets,t4_offsets,t5_offsets])
    for i,c in enumerate(coords):
        ax.text(*c, phase_nos[i], fontsize=10, ha='center', va='center',
                fontfamily="DejaVu Sans", zorder=999)
            
    ax.set_xticklabels(())
    #ax.set_xlabel('PC1')
    ax.set_yticklabels(())
    #ax.set_ylabel('PC2')
    ax.set_zticklabels(())
    #ax.set_zlabel('PC3')
    
    #ax.set_rasterized(True)


#%%
fits = analysis2ChoiceIntro.getTrajectories(endoDataPath)
train_fits = fits.loc['train']
test_fits = fits.loc['test']

ax = layout.axes['trajectory1']['axis']
plot3D(train_fits, test_fits.loc[test_fits.trialType.str.slice(0,4) == 'pR2C'],
       290, 135, lims=(-2.5,2.5),
       a1_offsets=[0,.3,0], a2_offsets=[.4,-.5,0], a3_offsets=[-.3,-.3,0],
       t1_offsets=[-.35,.35,0], t2_offsets=[.2,.4,0], t3_offsets=[.3,-.3,0],
       t4_offsets=[-.45,-.2,0], t5_offsets=[-.5,0,0], ax=ax)
ax.text(0,2.5,2.5, 'PC1', fontsize=7, ha='center', va='top',
        bbox=dict(facecolor='w', alpha=.75, pad=0))
ax.text(-2.5,0,2.5, 'PC2', fontsize=7, ha='center', va='top',
        bbox=dict(facecolor='w', alpha=.75, pad=0))
ax.text(-2.5,-2.5,0, 'PC3', fontsize=7, ha='center', va='center',
        bbox=dict(facecolor='w', alpha=.75, pad=0))

ax = layout.axes['trajectory2']['axis']
plot3D(train_fits, test_fits.loc[test_fits.trialType.str.slice(0,4) == 'pR2C'],
       355, 180, lims=(-2.5,2.5),
       a1_offsets=[0,0,.65], a2_offsets=[0,.1,-.28], a3_offsets=[0,-.35,.2],
       t1_offsets=[0,-.3,.4], t2_offsets=[.1,.5,0], t3_offsets=[0,-.43,0],
       t4_offsets=[0,-.4,.2], t5_offsets=[0,0,-.38], ax=ax)
ax.text(0,2.5,2.5, 'PC1', fontsize=7, ha='center', va='center',
        bbox=dict(facecolor='w', alpha=.75, pad=0))
ax.text(-2.5,0,2.5, 'PC2', fontsize=7, ha='center', va='top',
        bbox=dict(facecolor='w', alpha=.75, pad=0))
ax.text(-2.5,-2.5,0, 'PC3', fontsize=7, ha='center', va='top',
        bbox=dict(facecolor='w', alpha=.75, pad=0))


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)
subprocess.check_call(['inkscape', '-f', outputFolder / svgName,
                                   '-A', outputFolder / (svgName[:-3]+'pdf')])