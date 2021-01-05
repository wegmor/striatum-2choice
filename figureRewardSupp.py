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
trialsDf, tuningsDf, activityDf = analysisRewardSupp.getOutcomeResponseData(endoDataPath)

lRewTuned = tuningsDf.loc[tuningsDf.leftReward==1,'neuron']
rRewTuned = tuningsDf.loc[tuningsDf.rightReward==1, 'neuron']
lTrials = trialsDf.loc[trialsDf.side == 'L', ['trial','rSize']]
rTrials = trialsDf.loc[trialsDf.side == 'R', ['trial','rSize']]
rTrials = rTrials.loc[rTrials.trial != 'a2a_6043_190126_4817'] # huge artifact in trial

# compute left trial averages / reward size for left win-stay neurons
lt_avg_trial = activityDf.query('trial in @lTrials.trial & neuron in @lRewTuned').copy()
lt_avg_trial = pd.DataFrame(lt_avg_trial.iloc[:,7:].mean(axis=1), columns=['f']).reset_index() # drop delay period and average trials
lt_avg_trial = lt_avg_trial.merge(lTrials, on='trial', how='left')
lt_avg_trial = lt_avg_trial.groupby(['rSize','neuron']).mean().unstack('rSize').f
lt_avg_trial['genotype'] = lt_avg_trial.index.map(lambda n: n.split('_')[0])
lt_avg_trial = lt_avg_trial.reset_index().set_index(['genotype','neuron'])

# compute right trial averages / reward size for right win-stay neurons
rt_avg_trial = activityDf.query('trial in @rTrials.trial & neuron in @rRewTuned').copy()
rt_avg_trial = pd.DataFrame(rt_avg_trial.iloc[:,7:].mean(axis=1), columns=['f']).reset_index() # drop delay period and average
rt_avg_trial = rt_avg_trial.merge(rTrials, on='trial', how='left')
rt_avg_trial = rt_avg_trial.groupby(['rSize','neuron']).mean().unstack('rSize').f
rt_avg_trial['genotype'] = rt_avg_trial.index.map(lambda n: n.split('_')[0])
rt_avg_trial = rt_avg_trial.reset_index().set_index(['genotype','neuron'])

# compute left trial frame-by-frame averages / reward size for left win-stay neurons
lt_avg_frame = activityDf.query('trial in @lTrials.trial & neuron in @lRewTuned').copy()
lt_avg_frame = lt_avg_frame.reset_index().merge(lTrials, on='trial', how='left').set_index(['trial','neuron','rSize'])
lt_avg_frame = lt_avg_frame.groupby(['neuron','rSize']).mean()
lt_avg_frame['genotype'] = lt_avg_frame.index.get_level_values(0).map(lambda n: n.split('_')[0])
lt_avg_frame = lt_avg_frame.reset_index().set_index(['genotype','rSize','neuron'])

# compute left trial frame-by-frame averages / reward size for right win-stay neurons
rt_avg_frame = activityDf.query('trial in @rTrials.trial & neuron in @rRewTuned').copy()
rt_avg_frame = rt_avg_frame.reset_index().merge(rTrials, on='trial', how='left').set_index(['trial','neuron','rSize'])
rt_avg_frame = rt_avg_frame.groupby(['neuron','rSize']).mean()
rt_avg_frame['genotype'] = rt_avg_frame.index.get_level_values(0).map(lambda n: n.split('_')[0])
rt_avg_frame = rt_avg_frame.reset_index().set_index(['genotype','rSize','neuron'])


#%%
for genotype in ['d1','a2a','oprm1']:
    for (side, df) in zip(('L','R'), (lt_avg_trial.loc[genotype,[0,1,2]], rt_avg_trial.loc[genotype,[0,1,2]])):
        ax = layout.axes['{}_pointplot_{}'.format(genotype, {'L':'lt','R':'rt'}[side])]

        sns.stripplot(data=(df[[0,2]] - df[[1]].values),
                      ax=ax, color=style.getColor('p'+side),
                      size=1.5, alpha=.5, jitter=.32, linewidth=0, zorder=-1,
                      clip_on=True)
        means = (df - df[[1]].values).mean()
        sems = (df - df[[1]].values).sem()
        ax.errorbar([0,.5,1], means.loc[[0,1,2]], yerr=sems.loc[[0,1,2]],
                    fmt='ko-', markersize=3.2, markeredgewidth=0, zorder=1)
        
        ax.axhline(0, ls='--', zorder=0, c='k', alpha=.35, lw=mpl.rcParams['axes.linewidth'])
        ax.axvline(.5, ls='--', zorder=0, c='k', alpha=.35, lw=mpl.rcParams['axes.linewidth'])
        
        # ax.text(.975, .99, 'n={}'.format(df.shape[0]), fontsize=6, ha='right',
        #         va='top', transform=ax.transAxes)

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
for genotype in ['d1','a2a','oprm1']:
    for (side, df) in zip(('L','R'), (lt_avg_frame.loc[genotype], rt_avg_frame.loc[genotype])):
        ax = layout.axes['{}_trace_{}'.format(genotype, {'L':'lt','R':'rt'}[side])]
        
        df = df.groupby('rSize').agg(['mean','sem']).loc[:,:36].copy()
        df.columns = df.columns.reorder_levels((1,0))
        
        ax.axvline(-.5, ls='--', c='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
        ax.axvline(6.5, ls='--', c='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
        ax.axhline(0, ls='--', c='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
        
        ax.fill_between(np.arange(36),
                        df.loc[0,'mean'][:36]+df.loc[0,'sem'][:36],
                        df.loc[0,'mean'][:36]-df.loc[0,'sem'][:36],
                        alpha=.25, lw=0, clip_on=False, zorder=-3,
                        color=style.getColor('o!'))
        ax.plot(df.loc[0,'mean'][:36], color=style.getColor('o!'), alpha=.8,
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
        
        ax.set_ylim((-.25,.25))
        ax.set_xlim((-3.5,36.5))
        ax.set_yticks((-.25,0,.25))
        ax.set_xticks((-.5,6.5,16.5,26.5,36.5))
        ax.set_xticklabels(('','',.5,1.,1.5), rotation=0)
        if not (genotype == 'd1' and side == 'L'):
            ax.set_yticklabels(())
        else:
            ax.set_ylabel('sd')
            ax.legend(bbox_to_anchor=(1,.85), loc='center right')
        
        sns.despine(ax=ax, trim=False)


#%%
pieDf = tuningsDf[['leftReward','rightReward']].stack().reset_index()
pieDf.columns = list(pieDf.columns[:-2]) + ['phase','tuning']
pieDf = pd.DataFrame(pieDf.groupby(['genotype','phase','tuning']).size(),
                     columns=['noNeurons'])[::-1]
pieDf['label'] = pieDf.index.get_level_values('tuning').map({-1:'lose-switch',0:'none',1:'win-stay'}).values

for (gt,phase), df in pieDf.groupby(['genotype','phase']):
    ax = layout.axes['{}_pie_{}'.format(gt, {'leftReward':'lt', 'rightReward':'rt'}[phase])]['axis']
    ax.pie(df.noNeurons, #labels=df.label,
           colors=df.label.map({'win-stay':style.getColor({'leftReward':'pL2C','rightReward':'pR2C'}[phase]),
                                'lose-switch':[0,0,0,.1],
                                'none':[0,0,0,.1]}),
           explode=[.3,0,0])
    ax.text(1.3, .6, 'n={}\n({:.1%})'.format(df.loc[gt,phase,1].noNeurons,
                                        df.loc[gt,phase,1].noNeurons/df.noNeurons.sum()),
            transform=ax.transAxes, ha='left', va='center', fontsize=6)


#%%  
def _compAvgs(neuronsDf, trialsDf, activityDf):
    trialsDf = trialsDf.copy()
    # don't consider reward size (single or double)
    trialsDf['rSize'] = trialsDf.rSize.clip(0,1)
    trialsDf['prevRSize'] = trialsDf.prevRSize.clip(0,1)
    
    trialsAvg = activityDf.query('trial in @trialsDf.trial & neuron in @neuronsDf').copy()
    trialsAvg = pd.DataFrame(trialsAvg.iloc[:,7:].mean(axis=1), columns=['f']).reset_index() # drop delay period and average trials
    trialsAvg = trialsAvg.merge(trialsDf, on='trial', how='left')
    # group [(0,0), (0,1), ...] means - overall mean
    trialsAvg = trialsAvg.groupby(['neuron','prevRSize','rSize']).mean() - trialsAvg.groupby(['neuron']).mean()
    trialsAvg = trialsAvg.unstack(['prevRSize','rSize']).f
    trialsAvg['genotype'] = trialsAvg.index.map(lambda n: n.split('_')[0])
    trialsAvg = trialsAvg.reset_index().set_index(['genotype','neuron'])
    
    return trialsAvg
    
def bootstrap(data, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(data), size=len(data), replace=True)
        avgs.append(np.average(data.iloc[idx], axis=0, weights=weights.iloc[idx]))
    return(np.std(avgs, axis=0))


#%%
lWinTuned = tuningsDf.loc[tuningsDf.leftReward==1,'neuron']
rWinTuned = tuningsDf.loc[tuningsDf.rightReward==1,'neuron']
lLoseTuned = tuningsDf.loc[tuningsDf.leftReward==-1,'neuron']
rLoseTuned = tuningsDf.loc[tuningsDf.rightReward==-1,'neuron']
lTrials = trialsDf.loc[(trialsDf.side == 'L') & (trialsDf.prevSide == 'L'),
                       ['trial','rSize','prevRSize','value','prevValue']]
rTrials = trialsDf.loc[(trialsDf.side == 'R') & (trialsDf.prevSide == 'R'),
                       ['trial','rSize','prevRSize','value','prevValue']]
rTrials = rTrials.loc[rTrials.trial != 'a2a_6043_190126_4817']
lWinAvgs = _compAvgs(lWinTuned, lTrials, activityDf)
rWinAvgs = _compAvgs(rWinTuned, rTrials, activityDf)
lLoseAvgs = _compAvgs(lLoseTuned, lTrials, activityDf)
rLoseAvgs = _compAvgs(rLoseTuned, rTrials, activityDf)
#%%
order = [(0,0),(1,0),(0,1),(1,1)]
for genotype in ['d1','a2a','oprm1']:
    for side in ['L','R']:
        ax = layout.axes['{}{}ByOutcome'.format(genotype, side)]['axis']
        
        winAvgs = (lWinAvgs if side == 'L' else rWinAvgs).loc[genotype]
        winSess = winAvgs.index.map(lambda s: '_'.join(s.split('_')[:-1]))
        winNeurons = winAvgs.groupby(winSess).size() # no. neurons by session
        winAvgs = winAvgs.groupby(winSess).mean() # average by session
        
        loseAvgs = (lLoseAvgs if side == 'L' else rLoseAvgs).loc[genotype]
        loseSess = loseAvgs.index.map(lambda s: '_'.join(s.split('_')[:-1]))
        loseNeurons = loseAvgs.groupby(loseSess).size()
        loseAvgs = loseAvgs.groupby(loseSess).mean()

        wAvg = np.average(winAvgs[order], axis=0, weights=winNeurons)
        wSem = bootstrap(winAvgs[order], weights=winNeurons)
        ax.errorbar(np.arange(len(order)), wAvg, yerr=wSem, marker='o',
                    markersize=3.2, markeredgewidth=0, color=style.getColor('r.'))
        # sns.stripplot(data=winAvgs[order], color='b', zorder=-1,
        #               alpha=.05, size=1.5, jitter=.3, lw=0, ax=ax)
        # ax.plot(winAvgs[order].T.values, color='b', zorder=-1, alpha=.1,
        #         lw=mpl.rcParams['axes.linewidth'])
        for s in winAvgs.iterrows():
            ax.plot(s[1][order].values, color=style.getColor('r.'), zorder=-1, alpha=.15,
                    lw=winNeurons.loc[s[0]]/30, clip_on=False)
            
        wAvg = np.average(loseAvgs[order], axis=0, weights=loseNeurons)
        wSem = bootstrap(loseAvgs[order], weights=loseNeurons)
        ax.errorbar(np.arange(len(order)), wAvg, yerr=wSem, marker='o',
                    markersize=3.2, markeredgewidth=0, color=style.getColor('o!'))
        # sns.stripplot(data=loseAvgs[order], color='r', zorder=-1,
        #               alpha=.05, size=1.5, jitter=.3, lw=0, ax=ax)
        # ax.plot(loseAvgs[order].T.values, color='r', zorder=-1, alpha=.1,
        #         lw=mpl.rcParams['axes.linewidth'])
        for s in loseAvgs.iterrows():
            ax.plot(s[1][order].values, color=style.getColor('o!'), zorder=-1, alpha=.15,
                    lw=loseNeurons.loc[s[0]]/30, clip_on=False)
        
        ax.set_ylim((-.2,.2))
        ax.set_yticks((-.1,0,.1))
        if genotype == 'd1' and side == 'L':
            ax.set_ylabel('Δsd', labelpad=-2)
        else:
            ax.set_yticklabels(())
            ax.set_ylabel('')
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(())
        ax.set_xlabel('left port' if side == 'L' else 'right port', labelpad=6)
        color = sns.desaturate(style.getColor('p{}2C'.format(side)), .7)
        ax.xaxis.label.set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.tick_params(axis='x', colors=color)
        ax.axhline(0, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'])
        sns.despine(ax=ax, trim=True)


#%%
valueDf = trialsDf.loc[trialsDf.side == trialsDf.prevSide].copy()
valueDf['dValue'] = valueDf.value - valueDf.prevValue
valueDf['rSize'] = valueDf.rSize.clip(0,1)
valueDf['prevRSize'] = valueDf.prevRSize.clip(0,1)
valueDf = valueDf.groupby(['side','prevRSize','rSize','genotype','animal'])[['value','dValue']].mean()
##%%
vax = layout.axes['avByOutcome']['axis']
dax = layout.axes['davByOutcome']['axis']
order = [(0,0),(1,0),(0,1),(1,1)]

for side in ['L','R']:
    data = valueDf.loc[side].unstack(['prevRSize','rSize'])['value'][order] * (-1 if side == 'L' else 1)
    x = np.arange(len(order)) #np.arange(0, *((4,1) if side == 'R' else (-4,-1)))
    vax.plot(x, data.values.T, color=style.getColor('p{}2C'.format(side)),
             alpha=.2, lw=mpl.rcParams['axes.linewidth'])
    vax.errorbar(x, y=data.mean(), yerr=data.sem(), color=style.getColor('p{}2C'.format(side)),
                 label={'L':'left port', 'R':'right port'}[side])
vax.set_ylim((0,5))
vax.set_yticks(np.arange(6))
vax.set_ylabel('action value*')
vax.set_xlabel('outcomes', labelpad=6)
vax.set_xticks(np.arange(len(order)))
vax.set_xticklabels(())
vax.legend(bbox_to_anchor=(0.05,1), loc='upper left')
sns.despine(ax=vax, trim=True)

for side in ['L','R']:
    data = valueDf.loc[side].unstack(['prevRSize','rSize'])['dValue'][order] * (-1 if side == 'L' else 1)
    x = np.arange(len(order)) #np.arange(0, *((4,1) if side == 'R' else (-4,-1)))
    dax.plot(x, data.values.T, color=style.getColor('p{}2C'.format(side)),
             alpha=.2, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    dax.errorbar(x, y=data.mean(), yerr=data.sem(), color=style.getColor('p{}2C'.format(side)))
dax.set_ylim((-1.5,1.5))
dax.set_yticks((-1,0,1))
dax.set_yticks((-.5,.5), minor=True)
dax.axhline(0, ls=':', color='k', alpha=1, lw=mpl.rcParams['axes.linewidth'])
dax.set_ylabel('Δaction value*')
dax.set_xlabel('outcomes', labelpad=6)
dax.set_xticks(np.arange(len(order)))
dax.set_xticklabels(())
sns.despine(ax=dax, trim=True)


layout.insert_figures('plots')
layout.write_svg(outputFolder / "rewardSupp.svg")
