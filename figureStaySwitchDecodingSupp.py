#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:10:35 2019

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
import analysisStaySwitchDecoding
import analysisStaySwitchDecodingSupp
import analysisOftVs2Choice
from utils import fancyViz
from utils import readSessions
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
#import subprocess
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
    
    for action in ['dL2C','mL2C','pC2L','mC2L','dR2C','mR2C','pC2R','mC2R']:
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

cachedDataPath = cacheFolder / 'staySwitchAUC.pkl'
if cachedDataPath.is_file():
    staySwitchAUC = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath)
    staySwitchAUC.to_pickle(cachedDataPath)

cachedDataPath = cacheFolder / 'staySwitchAUC_shuffled.pkl'
if cachedDataPath.is_file():
    staySwitchAUC_shuffled = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC_shuffled = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath,
                                                                           n_shuffles=1,
                                                                           on_shuffled=True)
    staySwitchAUC_shuffled.to_pickle(cachedDataPath)
    
cachedDataPath = cacheFolder / "staySwitchAcrossDays.pkl"
if cachedDataPath.is_file():
    decodingAcrossDays = pd.read_pickle(cachedDataPath)
else:
    decodingAcrossDays = analysisStaySwitchDecoding.decodeStaySwitchAcrossDays(endoDataPath, alignmentDataPath)
    decodingAcrossDays.to_pickle(cachedDataPath)

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
svgName = 'staySwitchDecodingSupp.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName, dpi=600)
layout.make_mplfigures()


#%%
staySwitch_shuffled = staySwitchAUC_shuffled.copy()
staySwitch_shuffled['genotype'] = 'shuffled'
staySwitch = pd.concat([staySwitchAUC, staySwitch_shuffled])
palette = {gt: style.getColor(gt) for gt in ['d1','a2a','oprm1','shuffled']}

# plot kde
for a, adata in staySwitch.groupby('action'):
    if a.startswith('pL2C') or a.startswith('pR2C'): continue
    ax = layout.axes['{}_auc_kde'.format(a)]['axis']
    
    for gt, agdata in adata.groupby('genotype'):
#        ax.hist(agdata['auc'], bins=np.arange(-1,1.1,.1), histtype='step',
#                color=style.getColor(gt), label=gt, density=True,
#                lw=2, alpha=.8)
        sns.distplot(agdata['auc'], bins=np.arange(-1,1.1,.1),
                     ax=ax, color=style.getColor(gt), hist=False,
                     kde_kws={'clip_on':False, 'alpha':.75,
                              'zorder':-1 if gt == 'shuffled' else 1})
    
    ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
    ax.set_ylim((0,4))
    ax.set_yticks((1,3), minor=True)
    ax.set_yticks((0,2,4))
    ax.set_yticklabels(())
    ax.set_ylabel('')
    if a == 'mL2C':
        ax.set_ylabel('density')
        ax.set_yticklabels(ax.get_yticks())
    ax.set_xlim((-1,1))
    ax.set_xticks(())
    #ax.set_xticks((-1,0,1))
    #ax.set_xticklabels((-1,0,1))
    #ax.set_xticks((-.5,.5), minor=True)
    #ax.set_xlabel('selectivity score')
    ax.set_xlabel('')
    sns.despine(bottom=True, trim=True, ax=ax)
       
    ax = layout.axes['{}_auc_bp'.format(a)]['axis']

    sns.boxplot('auc', 'genotype', data=adata, ax=ax, 
                palette=palette, saturation=.85, showcaps=False, showfliers=False,
                boxprops={'alpha':0.75, 'lw':0, 'zorder':-99, 'clip_on':False}, 
                width=.75, whiskerprops={'c':'k','zorder':99, 'clip_on':False},
                medianprops={'c':'k','zorder':99, 'clip_on':False},
                order=['d1','a2a','oprm1','shuffled'][::-1])
    
    ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
    ax.set_xlim((-1,1))
    ax.set_ylim((-.75,3.75))
    ax.set_xticks((-1,0,1))
    #ax.set_xticklabels((-1,'',1))
    ax.set_xticks((-.5,.5), minor=True)
    ax.set_xlabel('')
    ax.set_yticks(())
    ax.set_ylabel('')
    sns.despine(left=True, trim=True, ax=ax)
        
axt = layout.axes['auc_legend']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.75,
                                 label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1',
                                        'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']
                  ]
axt.legend(handles=legend_elements, ncol=4, loc='center',
           mode='expand')
axt.set_title('selectivity score', fontsize=7)
axt.axis('off')


#%%
#staySwitchAUC['sign'] = (staySwitchAUC.pct > .995).astype('int') - (staySwitchAUC.pct < .005).astype('int')
#cmap = (mpl.colors.LinearSegmentedColormap
#                  .from_list('cmap', [sns.color_palette('bright')[4],
#                                      (.9,.9,.9),
#                                      sns.color_palette('bright')[2]]))
#
#df = staySwitchAUC.query('sign in [1,-1]').copy()
#    
#for action, pop_df in df.groupby('action'):
#    if action.startswith('pL2C') or action.startswith('pR2C'): continue
#    
#    axs = [layout.axes['{}_avg'.format(action+tt)]['axis']
#               for tt in ['r.','o.','o!']]
#    
#    if action == 'mL2C':
#        cax = layout.axes['colorbar']['axis']
#        analysisStaySwitchDecoding.drawPopAverageFV('data/endoData_2019.hdf', pop_df, axs, cax,
#                                                    auc_weigh=True, saturation=.2,                         #                                                    smoothing=5.5, cmap=cmap)
#    else:
#        analysisStaySwitchDecoding.drawPopAverageFV('data/endoData_2019.hdf', pop_df, axs,
#                                                    auc_weigh=True, saturation=.2,                         #                                                    smoothing=5.5, cmap=cmap)


#%%
prob_value_df = (P.set_index(['shuffled','genotype','animal','date','label','actionNo'])
                  .loc[False, ['action','o!','r.','noNeurons']])
prob_value_df['value'] = (actionValues.set_index(['genotype','animal','date','label','actionNo'])
                                      .value)
prob_value_df = prob_value_df.reset_index()
prob_value_df['stay'] = prob_value_df.label.str.endswith('.').astype('int')

  
##%%
for p, actions in enumerate((['pC2L','pC2R'],['mC2L','mC2R'],['dL2C','dR2C'])):
    data = prob_value_df.query('action in @actions').dropna().copy()
    data = data.loc[data.label.str.contains('o!$|o\.$|r\.$')]
    
    for (gt,label), gdata in data.groupby(['genotype','action']):
        ax = layout.axes['{}_value_ost_{}'.format(gt,p+1)]['axis']
        
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
        
        ax.axhline(.5, ls=':', c='k', alpha=.35, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
        ax.axvline(0, ls=':', c='k', alpha=.35, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
        
        ax.set_ylim((0,1))
        ax.set_xlim((-5,5))
        ax.set_xticks((-5,0,5))
        ax.set_xticklabels([])
        #ax.invert_xaxis()
        if gt == 'a2a':
            ax.set_xlabel('action value')
            ax.set_xticklabels([-5,0,5])
        ax.set_yticks((0,.5,1))
        if (gt == 'd1') & (p == 0):
            ax.set_yticklabels((0,50,100))
            #ax.set_ylabel('SVM prediction\nP(win-stay)')
        else:
            ax.set_yticklabels(())
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))
        sns.despine(ax=ax)    


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


for p, actions in enumerate((['pC2L','pC2R'],['mC2L','mC2R'],['dL2C','dR2C'])):
    data = prob_value_df.query('action in @actions').dropna().copy()
    data = data.loc[data.label.str.contains('o!$|o\.$|r\.$')]
    
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
        ax = layout.axes['{}_{}_corr_{}'.format(gt,tt,p+1)]['axis']

        wAvg = analysisStaySwitchDecoding.wAvg(cs, 'correlation', 'noNeurons')
        wSem = analysisStaySwitchDecoding.bootstrap(cs, 'correlation', 'noNeurons')
        r_wAvg = analysisStaySwitchDecoding.wAvg(cs, 'rand_correlation', 'noNeurons')
        r_wSem = analysisStaySwitchDecoding.bootstrap(cs, 'rand_correlation', 'noNeurons')

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

        ax.axhline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])

        ax.set_ylim((0,.5))    
        ax.set_xlim((-.35,1.35))
        if tt == 'r.':
            ax.set_xticks(())
            ax.set_yticks((0,.5))
            ax.set_yticklabels(())
            ax.set_yticks((.25,), minor=True)
            if gt == 'd1' and p == 0:
                ax.set_ylabel('r')
                ax.set_yticklabels((.0,.5))
            sns.despine(ax=ax, bottom=True, trim=True)
        elif tt == 'o.':
            ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[gt])
            ax.set_axis_off()
        else:
            ax.set_axis_off()

            
axt = layout.axes['svm_legend']['axis']
legend_elements = [mlines.Line2D([0], [0], marker='<', color='k', markersize=2.8,
                                     markeredgewidth=0, label='(left) choice', lw=0),
                   mpatches.Patch(color=style.getColor('r.'), alpha=1,
                                  label='win-stay'),                          
                   mpatches.Patch(color=style.getColor('o.'), alpha=1,
                                  label='lose-stay'),
                   mpatches.Patch(color=style.getColor('o!'), alpha=1,
                                  label='lose-switch'),
                   mlines.Line2D([0], [0], marker='o', markersize=3.2, markerfacecolor='w',
                                 markeredgewidth=.8, color='k', ls='-',
                                 label='action values shifted')
                  ]
axt.legend(handles=legend_elements, ncol=len(legend_elements), loc='center',
           mode='expand')
axt.axis('off')

    
#%%
def bootstrapSEM(values, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(values), len(values), replace=True)
        avgs.append(np.average(values.iloc[idx], weights=weights.iloc[idx]))
    return np.std(avgs)

fromDate = pd.to_datetime(decodingAcrossDays.fromDate, format="%y%m%d")
toDate = pd.to_datetime(decodingAcrossDays.toDate, format="%y%m%d")
td = (toDate - fromDate).dt.days
decodingAcrossDays["dayDifference"] = td

for label in ('mL2C','pC2L','mC2L','dL2C','mR2C','pC2R','mC2R','dR2C'):
    selection = decodingAcrossDays[decodingAcrossDays.label == label]
    for i,l,h in ((0,1,100),):#((0,1,3), (1,4,13), (2,14,100)):
        g = (selection.query("dayDifference >= {} & dayDifference <= {}".format(l,h))
                      .groupby(["animal", "fromDate", "toDate"]))

        perAnimal = g.mean()[['nNeurons', 'sameDayScore', 'nextDayScore',
                              'sameDayShuffled', 'nextDayShuffled']]
        perAnimal["genotype"] = g.genotype.first()


        scaledScore = perAnimal[['sameDayScore', 'nextDayScore',
                                 'sameDayShuffled', 'nextDayShuffled']] * \
                      perAnimal.nNeurons[:,np.newaxis]
        
        perGenotype = scaledScore.groupby(perAnimal.genotype).sum()
        totalNeurons = perAnimal.groupby('genotype').nNeurons.sum()
        perGenotype /= totalNeurons[:,np.newaxis]

        shuffleScore = scaledScore[['sameDayShuffled','nextDayShuffled']].sum() / \
                       perAnimal.nNeurons.sum()

        plt.sca(layout.axes["decodingAcrossDays_{}".format(label)]["axis"])
        
        # linewidth fails in pdf export?
        #for r in perAnimal.itertuples():
        #    plt.plot([0,1], [r.sameDayScore, r.nextDayScore],
        #             lw=mpl.rcParams['axes.linewidth']*r.nNeurons/400.0, alpha=.25,
        #             clip_on=False, zorder=-99, color=style.getColor(r.genotype))
        for r in perGenotype.itertuples():
            gt = r.Index
            animalsWithGt = perAnimal.query("genotype == '{}'".format(gt))
            sameDaySEM = bootstrapSEM(animalsWithGt.sameDayScore, animalsWithGt.nNeurons)
            nextDaySEM = bootstrapSEM(animalsWithGt.nextDayScore, animalsWithGt.nNeurons)
            plt.errorbar([0,1], [r.sameDayScore, r.nextDayScore], [sameDaySEM, nextDaySEM],
                         lw=style.lw(), c=style.getColor(gt))
        
        sameDayShuffledSEM = bootstrapSEM(perAnimal.sameDayShuffled, perAnimal.nNeurons)
        nextDayShuffledSEM = bootstrapSEM(perAnimal.nextDayShuffled, perAnimal.nNeurons)
        plt.errorbar([0,1], [shuffleScore.sameDayShuffled, shuffleScore.nextDayShuffled],
                     [sameDayShuffledSEM, nextDayShuffledSEM],
                     lw=style.lw(), c=style.getColor("shuffled"))

        plt.axhline(0.5, lw=mpl.rcParams['axes.linewidth'], c='k', alpha=.5,
                    ls=':', clip_on=False)
        
        
        plt.ylim(.5,1)
        plt.xlim(-0.25, 1.25)
        plt.xticks([])
        #plt.xlabel(("1-3", "4-13", "14+")[i] + "\ndays", labelpad=7, fontsize=6)
        if label in ['pC2L','pC2R']:
            plt.yticks(np.linspace(.5,1,3), np.linspace(50,100,3,dtype=np.int64))
            #plt.ylabel("decoding accuracy (%)")
        else:
            plt.yticks(np.linspace(.5,1,3), [])
        plt.axhline(0.5, color='k', lw=0.5, alpha=0.5, ls=":")
        sns.despine(ax=plt.gca(), left=(i!=0), bottom=True)
        plt.gca().spines['left'].set_color(style.getColor(label))
        plt.gca().spines['left'].set_linewidth(1.25)
        plt.gca().tick_params(axis='y', colors=style.getColor(label), labelcolor='k',
                              width=1.25)

axt = layout.axes['across_days_legend']['axis']
legend_elements = [mpl.lines.Line2D([0],[0], color=style.getColor(gt),
                                    label={'oprm1':'Oprm1','a2a':'A2A','d1':'D1',
                                           'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']]
axt.legend(handles=legend_elements, loc='center', ncol=4, mode='expand')
axt.axis('off')


#%% value coding neuron example plots
def avRegPlot(means, phase='mS2C', ax=None):
    inclLabels = analysisStaySwitchDecodingSupp.getPhaseLabels(phase)
    data = means.loc[means.label.isin(inclLabels)]
    for l, ldata in data.groupby('label'):
        ax.scatter(ldata['value'], ldata['trialMean'],
                   facecolor='w', edgecolor=style.getColor(l[-2:]),
                   #marker='<' if 'L' in l else '>',
                   marker='o', alpha=.25, s=3.5, lw=.5, clip_on=True)
        ax.errorbar(ldata['value'].mean(), ldata['trialMean'].mean(),
                    xerr=ldata['value'].sem(), yerr=ldata['trialMean'].sem(),
                    color=sns.desaturate(style.getColor(l[-2:]),.8),
                    #marker='<' if 'L' in l else '>',
                    marker='o', ms=3, clip_on=False)
    sns.regplot('value', 'trialMean', data=data, fit_reg=True, scatter=False,
                ax=ax, color='k', ci=False, line_kws={'zorder':-99, 'lw':.5})
    
    
def avAvgTracePlot(wins, phase='mS2C', compression=40, ax=None):
    inclLabels = analysisStaySwitchDecodingSupp.getPhaseLabels(phase)
    for l,ldata in wins.loc[wins.label.isin(inclLabels)].groupby('label'):
        x = np.array(ldata['frameNo'].columns.values / compression + ldata['value'].mean(),
                     dtype='float')
        x_offset = -(len(x) // 2) / compression
        y = ldata['frameNo'].mean().values
        y[ldata['frameNo'].notna().sum(axis=0) < 20] = np.nan
        sem = ldata['frameNo'].sem().values
        sem[ldata['frameNo'].notna().sum(axis=0) < 20] = np.nan
        ax.fill_between(x + x_offset, y-sem, y+sem, clip_on=False,
                        color=style.getColor(l[-2:]), lw=0, alpha=.5)
        ax.plot(x + x_offset, y, color=style.getColor(l[-2:]), alpha=.8,
                clip_on=False)
        ax.axvline(ldata['value'].mean(), ls=':', color='k', alpha=1,
                   lw=mpl.rcParams['axes.linewidth'])
        
#%%
exGenotype, exAnimal, exDate = 'oprm1', '5703', '190130'
exNeurons = [100,26]

#%% plot F8 schematic
s = next(readSessions.findSessions(endoDataPath, genotype=exGenotype,
                                   animal=exAnimal, date=exDate, task='2choice'))
traces = s.readDeconvolvedTraces(rScore=True)
lfa = s.labelFrameActions(reward='fullTrial', switch=True).set_index(traces.index)
for p,n in enumerate(exNeurons):
    trace = traces[n]  
    for trialType in ['r.','o.','o!']:
        axfv = layout.axes['f8_ex{}_{}'.format(p+1, trialType)]['axis']
        fv = fancyViz.SchematicIntensityPlot(s, splitReturns=False,
                                             linewidth=mpl.rcParams['axes.linewidth'],
                                             smoothing=7.5, saturation=2)
        fv.setMask(lfa.label.str.endswith(trialType).values)
        img = fv.draw(trace, ax=axfv)
        axfv.axis('on')
        axfv.set_xlabel({'r.':'win-stay','o.':'lose-stay','o!':'lose-switch'}[trialType],
                        color=style.getColor(trialType))
        axfv.set_xticks(()); axfv.set_yticks(())
        sns.despine(ax=axfv, left=True, bottom=True)

cbs = []
for c in [1,2]:
    cax = layout.axes['colorbar'+str(c)]['axis']
    cb = plt.colorbar(img, cax=cax, orientation='horizontal')
    cax.text(-2.05, -.3, '-2', ha='right', va='center', fontdict={'fontsize':6})
    cax.text(2.05, -.3, '2', ha='left', va='center', fontdict={'fontsize':6})
    cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})
    cax.axis('off')
    cbs.append(cb)
[cb.outline.set_visible(False) for cb in cbs]
        

#%% plot regression an average trace plots of examples
means = analysisStaySwitchDecodingSupp.getActionMeans(endoDataPath, exGenotype, exAnimal, exDate)
means = means.loc[means.neuron.isin(exNeurons)].set_index('actionNo').sort_index()
wins = analysisStaySwitchDecodingSupp.getActionWindows(endoDataPath, exGenotype, exAnimal, exDate)
wins = wins.loc[wins.neuron.isin(exNeurons)].set_index('actionNo').sort_index()
av = actionValues.set_index(['genotype','animal','date','actionNo']).sort_index().copy()

av = av.loc[(exGenotype, exAnimal, exDate)].copy()
means['value'] = av.value
wins['value'] = av.value
means = means.reset_index().set_index(['neuron','actionNo']).sort_index()
wins = wins.reset_index().set_index(['neuron','actionNo']).sort_index()

for p, neuron in enumerate(exNeurons):
    regAx = layout.axes['ac1_ex{}'.format(p+1)]
    avgAx = layout.axes['ac2_ex{}'.format(p+1)]
    avRegPlot(means.loc[neuron],phase='mR2C',ax=regAx)
    avAvgTracePlot(wins.loc[neuron],phase='mR2C',compression=15,ax=avgAx)
    
    for ax in [regAx, avgAx]:
        ax.set_ylim((-.75,6))
        ax.set_yticks((0,2,4,6))
        ax.set_yticks((1,3,5), minor=True)
        ax.set_xlim((-1,5))
        ax.set_ylabel('z-score')
    avgAx.set_xticks(np.arange(-1,6))
    avgAx.set_xticks(np.arange(-1,5,.5), minor=True)
    regAx.set_xticks(())
    avgAx.set_xlabel('action value')
    regAx.set_xlabel('')
    sns.despine(ax=avgAx, trim=False)
    sns.despine(ax=regAx, bottom=True, trim=True)


#%% plot movement trajectory and duration plots
chSess = next(readSessions.findSessions(endoDataPath, genotype=exGenotype,
                                        animal=exAnimal, date=exDate))
chTracking = analysisOftVs2Choice.getSmoothedTracking(endoDataPath, exGenotype,
                                                      exAnimal, exDate, '2choice')

ax0, ax1, ax2 = [layout.axes['{}TrajMap'.format(tt)]['axis'] for tt in ['rst','ost','osw']]

chTracking['bin'] = chTracking.actionProgress * 100 // 10

# trajectories
rst = chTracking.loc[chTracking.behavior.str.startswith('mR2Cr.')]
rst_mean = rst.groupby('bin').mean()
for actionNo, track in rst.groupby('actionNo'):
    analysisStaySwitchDecodingSupp.plot2CTrackingEvent(track, ax0, color=style.getColor('r.'),
                                                       alpha=.025)
analysisStaySwitchDecodingSupp.plot2CTrackingEvent(rst_mean, ax0, color='k', lw=.5, alpha=.5)

ost = chTracking.loc[chTracking.behavior.str.startswith('mR2Co.')]
ost_mean = ost.groupby('bin').mean()
for actionNo, track in ost.groupby('actionNo'):
    analysisStaySwitchDecodingSupp.plot2CTrackingEvent(track, ax1, color=style.getColor('o.'),
                                                       alpha=.025)
analysisStaySwitchDecodingSupp.plot2CTrackingEvent(ost_mean, ax1, color='k', lw=.5, alpha=.5)

osw = chTracking.loc[chTracking.behavior.str.startswith('mR2Co!')]
osw_mean = osw.groupby('bin').mean()
for actionNo, track in osw.groupby('actionNo'):
    analysisStaySwitchDecodingSupp.plot2CTrackingEvent(track, ax2, color=style.getColor('o!'),
                                                       alpha=.025)
analysisStaySwitchDecodingSupp.plot2CTrackingEvent(osw_mean, ax2, color='k', lw=.5, alpha=.5)

for ax in [ax0, ax1,ax2]:
    t = ax.transData
    t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(15/2, 15/2, 90) + t
    corners_x, corners_y = [0,0,15,15,0], [0,15,15,0,0]
    ax.plot(corners_x, corners_y, 'k', lw=0.5, transform=t)
    s = 15/7
    for y in s*np.array([1, 3, 5]):
        fancyViz.drawRoundedRect(ax, (15, y), s, s, [0, 0, s/4, s/4],
                                 fill=False, edgecolor="k", lw=mpl.rcParams['axes.linewidth'],
                                 transform=t)

ax2.set_xlabel('right to center turn\nmovement trajectories', labelpad=3)
ax0.set_ylabel('win-stay', color=style.getColor('r.'))
ax1.set_ylabel('lose-stay', color=style.getColor('o.'))
ax2.set_ylabel('lose-switch', color=style.getColor('o!'))
for ax in [ax0,ax1,ax2]:
    ax.set_aspect('equal')
    ax.set_xlim((-1, 16))
    ax.set_ylim((7.5, 18))
    ax.set_xticks(())
    ax.set_yticks(())
    ax.yaxis.set_label_coords(0, .35)
    sns.despine(top=True, left=True, right=True, bottom=True, ax=ax) 
    

# pairwise trajectory distance density plot
trajectories = chTracking.loc[chTracking.behavior.isin(['mR2Cr.','mR2Co.','mR2Co!'])]
df = analysisStaySwitchDecodingSupp.get2CTrajectoryDists(trajectories,
                                        '{}_{}_{}'.format(exGenotype, exAnimal, exDate))

axs = [layout.axes['{}TrajKde'.format(tt)]['axis'] for tt in ['rst',]]

pdict = {'rst':(('r.Xr.','o.Xr.','o!Xr.'),
                (style.getColor('r.'),style.getColor('o.'), style.getColor('o!'))),
         'ost':(('o.Xo.','o.Xr.','o!Xo.'),
                (style.getColor('o.'),style.getColor('r.'), style.getColor('o!'))),
         'osw':(('o!Xo!','o!Xr.','o!Xo.'),
                (style.getColor('o!'),style.getColor('r.'), style.getColor('o.')))}

for p,tt in enumerate(('rst',)):
    ax = axs[p]
    strs, cs = pdict[tt]
    
    sns.distplot(df.loc[df.trialTypes == strs[0],'distance'], hist=False, color=cs[0],
                 kde_kws={'clip_on':True, 'alpha':.75, 'cut':3, 'lw':.8}, ax=ax)
    sns.distplot(df.loc[df.trialTypes == strs[1],'distance'], hist=False, color=cs[1],
                 kde_kws={'clip_on':True, 'alpha':.75, 'cut':3, 'lw':.8}, ax=ax)
    sns.distplot(df.loc[df.trialTypes == strs[2],'distance'], hist=False, color=cs[2],
                 kde_kws={'clip_on':True, 'alpha':.75, 'cut':3, 'lw':.8}, ax=ax)
    
    ax.set_ylim((0,2))
    ax.set_yticks((0,1,2))
    ax.set_yticks((.5,1.5), minor=True)
    ax.set_yticklabels(())
    ax.set_ylabel('')
    ax.set_xlim((0,4))
    ax.set_xticks((0,2,4))
    ax.set_xticks((1,3), minor=True)
    ax.set_xticklabels(())
    ax.set_xlabel('')
    #if tt == 'osw':
    ax.set_xticklabels(ax.get_xticks())
    ax.set_xlabel('pairwise trajectory\ndistance (cm)')
    #if tt == 'ost':
    ax.set_yticklabels(ax.get_yticks())
    ax.set_ylabel('density')
    sns.despine(ax=ax, offset=.1)

legend_elements = [mpl.patches.Patch(color=color,
                                     label={'r.Xr.':'win-stay',
                                            'o.Xr.':'lose-stay',
                                            'o!Xr.':'lose-switch'}[tt]) 
                       for tt, color in zip(*pdict['rst'])]
lg = axs[0].legend(handles=legend_elements, ncol=1, title='win-stay vs ...',
                   bbox_to_anchor=(1,1), loc='upper right')
lg.get_title().set_fontsize(6)


# duration density plot
aDurations = chTracking.loc[chTracking.behavior.isin(['mR2Cr.','mR2Co.','mR2Co!'])]
aDurations = aDurations.groupby(['behavior','actionNo']).size() / 20.

ax = layout.axes['durationKde']['axis']
for action, ds in aDurations.groupby('behavior'):
    sns.distplot(ds, hist=False, color=style.getColor(action[-2:]),
                 kde_kws={'alpha':.75}, ax=ax)

ax.set_xlim((0,2))
ax.set_xticks((0,1,2))
ax.set_xticks((.5,1.5), minor=True)
ax.set_xlabel('turn duration (s)')
ax.set_ylim((0,4))
ax.set_yticks((0,2,4))
ax.set_yticks((1,3), minor=True)
ax.set_ylabel('density')
sns.despine(ax=ax)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)
#subprocess.check_call(['inkscape', '-f', outputFolder / svgName,
#                                   '-A', outputFolder / (svgName[:-3]+'pdf')])
