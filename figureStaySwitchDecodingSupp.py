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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import subprocess

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
    
    for action in ['dL2C','mL2C','pC2L','mC2L','dR2C','mR2C','pC2R','mC2R','pL2C','pR2C']:
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
    
cachedDataPath = cacheFolder / "staySwitchCrossDecoding.pkl"
if cachedDataPath.is_file():
    crossDecoding = pd.read_pickle(cachedDataPath)
else:
    crossDecoding, shuffleCross = (analysisStaySwitchDecoding
                                      .crossDecodeStaySwitch(endoDataPath))
    crossDecoding = crossDecoding.set_index(['genotype','animal','date',
                                            'testAction','trainAction'])
    shuffleCross = shuffleCross.set_index(['genotype','animal','date',
                                          'testAction','trainAction'])
    crossDecoding['accuracy_shuffle'] = shuffleCross.accuracy
    crossDecoding = (crossDecoding[['noNeurons','accuracy','accuracy_shuffle']]
                                  .reset_index())
    crossDecoding.to_pickle(cachedDataPath)
    
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


#%% plot matrices
cdMeans = analysisStaySwitchDecodingSupp.getStSwCodingDirectionRaster(endoDataPath)
cmap = (mpl.colors.LinearSegmentedColormap
           .from_list('wStLSw', [style.getColor('o!'), (1,1,1), style.getColor('r.')], 256))
cbar = mpl.colors.LinearSegmentedColormap.from_list('test',[style.getColor(c) for c in cdMeans.columns],
                                                    len(cdMeans.columns))

cax = layout.axes['CDlegend']['axis']
vMinMax = .1

for gt in ['d1','a2a']:
    ax = layout.axes[gt+'CD']['axis']
    if gt == 'd1':
        sns.heatmap(cdMeans.loc[gt].T, center=0, cmap=cmap, square=True, vmin=-vMinMax, vmax=vMinMax, ax=ax,
                    cbar_ax=cax, cbar_kws={'orientation':'horizontal', 'ticks':()})
    else:
        sns.heatmap(cdMeans.loc[gt].T, center=0, cmap=cmap, square=True, vmin=-vMinMax, vmax=vMinMax, ax=ax,
                    cbar=False)
    ax.vlines(np.arange(0,31,5), 0, 10, color='k', clip_on=False)
    ax.vlines(np.arange(0,31), 0, 10, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines(np.arange(1,10), 0, 30, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines([0,5,10], 0, 30, ls='-', color='k', clip_on=False)
    ax.set_yticks(())
    ax.set_xticks(())
    if gt == 'd1':
        ax.set_ylabel('coding direction', labelpad=18, fontsize=8)
    else:
        ax.set_ylabel('')
    ax.set_xlabel('trial phase', labelpad=26, fontsize=8)
    ax.set_title('{}'.format({'d1':'D1+', 'a2a':'A2A+'}[gt]), pad=6)#, loc='left')
    
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

cax.set_xlabel('CD (a.u.)', fontsize=6)
cax.text(-.025,.5,str(vMinMax), ha='right', va='center', transform=cax.transAxes,
         fontsize=6)
cax.text(1.025,.5,str(-vMinMax), ha='left', va='center', transform=cax.transAxes,
         fontsize=6)
#cax.tick_params(axis='x', length=0)
cax.invert_xaxis()


#%%
prob_value_df = (P.set_index(['shuffled','genotype','animal','date','label','actionNo'])
                  .loc[False, ['action','o!','r.','noNeurons']].copy())
prob_value_df['value'] = (actionValues.set_index(['genotype','animal','date','label','actionNo'])
                                      .value.shift(-1)) # update on delivery, not return movement
prob_value_df = prob_value_df.reset_index()
prob_value_df['stay'] = prob_value_df.label.str.endswith('.').astype('int')

  
##%%
for p, actions in enumerate((['pC2L','pC2R'],['mC2L','mC2R'],['dL2C','dR2C'],['pL2C','pR2C'])):
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


##%% svm prediction X value correlations
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


for p, actions in enumerate((['pC2L','pC2R'],['mC2L','mC2R'],['dL2C','dR2C'],['pL2C','pR2C'])):
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
            if gt == 'd1' and p in [0,2]:
                ax.set_ylabel('r')
                ax.set_yticklabels((.0,.5))
            sns.despine(ax=ax, bottom=True, trim=True)
        elif tt == 'o.':
            ax.set_title({'d1':'D1+','a2a':'A2A+','oprm1':'Oprm1+'}[gt])
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

for label in ('pL2C','mL2C','pC2L','mC2L','dL2C','pR2C','mR2C','pC2R','mC2R','dR2C'):
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
        
        # # linewidth fails in pdf export?
        # for r in perAnimal.itertuples():
        #     plt.plot([0,1], [r.sameDayScore, r.nextDayScore],
        #             lw=mpl.rcParams['axes.linewidth'],#*r.nNeurons/400.0,
        #             alpha=.25, clip_on=False, zorder=-99, color=style.getColor(r.genotype))
        for r in perGenotype.itertuples():
            gt = r.Index
            animalsWithGt = perAnimal.query("genotype == '{}'".format(gt))
            sameDaySEM = bootstrapSEM(animalsWithGt.sameDayScore, animalsWithGt.nNeurons)
            nextDaySEM = bootstrapSEM(animalsWithGt.nextDayScore, animalsWithGt.nNeurons)
            e = plt.errorbar([0,1], [r.sameDayScore, r.nextDayScore], [sameDaySEM, nextDaySEM],
                             lw=style.lw(), c=style.getColor(gt), clip_on=False)
            for b in e[1]:
                b.set_clip_on(False)
        
        sameDayShuffledSEM = bootstrapSEM(perAnimal.sameDayShuffled, perAnimal.nNeurons)
        nextDayShuffledSEM = bootstrapSEM(perAnimal.nextDayShuffled, perAnimal.nNeurons)
        e = plt.errorbar([0,1], [shuffleScore.sameDayShuffled, shuffleScore.nextDayShuffled],
                         [sameDayShuffledSEM, nextDayShuffledSEM],
                         lw=style.lw(), c=style.getColor("shuffled"), clip_on=False)
        for b in e[1]:
            b.set_clip_on(False)

        plt.axhline(0.5, lw=mpl.rcParams['axes.linewidth'], c='k', alpha=.5,
                    ls=':', clip_on=False)
        
        
        plt.ylim(.5,1)
        plt.xlim(-0.25, 1.25)
        plt.xticks([])
        #plt.xlabel(("1-3", "4-13", "14+")[i] + "\ndays", labelpad=7, fontsize=6)
        if label in ['pL2C','pR2C']:
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
                                    label={'oprm1':'Oprm1+','a2a':'A2A+','d1':'D1+',
                                           'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']]
axt.legend(handles=legend_elements, loc='center', ncol=2, mode='expand')
axt.axis('off')


#%%
acc = P.loc[P.label.str.contains('r\.$|o!$')].copy() # only use win-stay, lose-switch trials
acc = (acc.groupby(['genotype','animal','date','noNeurons','action', 'shuffled'])
          .apply(lambda sess: np.mean(sess.prediction == sess.label))
          .unstack('shuffled'))
acc = acc.rename(columns={False:'accuracy',True:'accuracy_shuffle'})
acc.reset_index(inplace=True)
acc = acc.rename(columns={'action':'trainAction'})
acc['testAction'] = acc.trainAction
crossDecoding = crossDecoding.append(acc, sort=True)

order = ['pL2C','mL2C','pC2L','mC2L','dL2C','pR2C','mR2C','pC2R','mC2R','dR2C']
crossDecoding['dAccuracy'] = crossDecoding['accuracy'] - crossDecoding['accuracy_shuffle']
difference = (crossDecoding.groupby(['genotype','trainAction','testAction'])
                           .apply(lambda g: np.average(g.dAccuracy, weights=g.noNeurons))
                           .unstack('testAction')) * 100

cax = layout.axes['crossDecodingLegend']['axis']
vMinMax = 50
for gt in ['d1','a2a','oprm1']:
    ax = layout.axes['{}CrossDecoding'.format(gt)]['axis']
    if gt == 'd1':
        sns.heatmap(difference.loc[gt].loc[order,order], center=0, cmap='RdYlBu_r',
                    cbar_ax=cax, cbar_kws={'orientation':'horizontal', 'ticks':()},
                    square=True, vmin=-vMinMax, vmax=vMinMax, ax=ax)
    else:
        sns.heatmap(difference.loc[gt].loc[order,order], center=0, cmap='RdYlBu_r',
                    cbar=False, square=True, vmin=-vMinMax, vmax=vMinMax, ax=ax)
    
    ax.vlines(np.arange(0,11,5), 0, 10, color='k', clip_on=False)
    ax.vlines(np.arange(0,11), 0, 10, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines(np.arange(1,10), 0, 10, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'],
              clip_on=False)
    ax.hlines([0,5,10], 0, 10, ls='-', color='k', clip_on=False)
    ax.set_yticks(())
    ax.set_xticks(())
    ax.set_ylabel('trained phase', labelpad=17, fontsize=7)
    ax.set_xlabel('predicted phase', labelpad=17, fontsize=7)
    ax.set_title({'a2a':'A2A+', 'd1':'D1+', 'oprm1':'Oprm1+'}[gt], pad=7)
    
    cbarYAx = layout.axes['{}CrossDecoding_ybar'.format(gt)]['axis']
    cbarXAx = layout.axes['{}CrossDecoding_xbar'.format(gt)]['axis']
    cbarYAx.pcolormesh(np.linspace(0,10,10)[:,np.newaxis], cmap=cbar)
    cbarXAx.pcolormesh(np.linspace(0,10,10)[np.newaxis,:], cmap=cbar)
    cbarYAx.hlines(np.arange(1,10), 0, 1, ls='-', color='k', lw=mpl.rcParams['axes.linewidth'])
    cbarYAx.hlines([0,5,10], -2, 1, ls='-', color='k', clip_on=False)
    cbarYAx.set_xlim((0,1))
    cbarXAx.vlines(np.arange(1,10), 0, 1, ls=':', color='k', lw=mpl.rcParams['axes.linewidth'])
    cbarXAx.vlines([5,], 1, -2, ls='-', color='k', clip_on=False)
    cbarXAx.vlines([0,10], 1, -4, ls='-', color='k', clip_on=False)
    cbarXAx.set_ylim((0,1))
    cbarYAx.axis('off')
    cbarYAx.invert_yaxis()
    cbarXAx.axis('off')

cax.set_xlabel('observed accuracy (%) - chance accuracy (%)', fontsize=6)
cax.text(-.025,.5,str(-vMinMax), ha='right', va='center', transform=cax.transAxes,
         fontsize=6)
cax.text(1.025,.5,str(vMinMax), ha='left', va='center', transform=cax.transAxes,
         fontsize=6)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)
subprocess.check_call(['inkscape', outputFolder / svgName,
                           '--export-pdf={}pdf'.format(outputFolder / svgName[:-3])])

