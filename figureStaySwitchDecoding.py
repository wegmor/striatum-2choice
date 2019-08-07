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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import pathlib
import figurefirst
import style
import analysisStaySwitchDecoding
import cmocean
from scipy.stats import ttest_1samp
#import statsmodels.formula.api as smf
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

cachedDataPaths = [cacheFolder / name for name in ['actionValues.pkl',
                                                   'logRegCoefficients.pkl',
                                                   'logRegDF.pkl']]
if np.all([path.is_file() for path in cachedDataPaths]):
    actionValues = pd.read_pickle(cachedDataPaths[0])
    logRegCoef = pd.read_pickle(cachedDataPaths[1])
    logRegDF = pd.read_pickle(cachedDataPaths[2])
else:
    actionValues, logRegCoef, logRegDF = analysisStaySwitchDecoding.getActionValues(endoDataPath)
    actionValues.to_pickle(cachedDataPaths[0])
    logRegCoef.to_pickle(cachedDataPaths[1])
    logRegDF.to_pickle(cachedDataPaths[2])
    
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
    

#%%
def bootstrapSEM(values, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(values), len(values), replace=True)
        avgs.append(np.average(values.iloc[idx], weights=weights.iloc[idx]))
    return np.std(avgs)
    
#accrossDays = accrossDays.rename(columns={"sameDayShuffled": "nextDayScore", "nextDayScore": "sameDayShuffled"})
fromDate = pd.to_datetime(decodingAcrossDays.fromDate, format="%y%m%d")
toDate = pd.to_datetime(decodingAcrossDays.toDate, format="%y%m%d")
td = (toDate - fromDate).dt.days
decodingAcrossDays["dayDifference"] = td

for label in ("pC2L", "pC2R", "mC2L", "mC2R", "mL2C", "mR2C"):
    selection = decodingAcrossDays[decodingAcrossDays.label == label]
    for i,l,h in ((0,1,3), (1,4,14), (2,14,100)):#(1,4,6), (2,7,14), (3,14,100)):
        g = selection.query("dayDifference >= {} & dayDifference <= {}".format(l,h)).groupby(["animal", "fromDate", "toDate"])

        perAnimal = g.mean()[['nNeurons', 'sameDayScore', 'nextDayScore', 'sameDayShuffled', 'nextDayShuffled']]
        perAnimal["genotype"] = g.genotype.first()


        scaledScore = perAnimal[['sameDayScore', 'nextDayScore']] * np.stack([perAnimal.nNeurons,
                                                                              perAnimal.nNeurons],
                                                                             axis=1)
        perGenotype = scaledScore.groupby(perAnimal.genotype).sum()
        totalNeurons = perAnimal.groupby('genotype').nNeurons.sum()
        perGenotype /= np.stack([totalNeurons,totalNeurons], axis=1)

        shuffleScore = perAnimal[['sameDayShuffled', 'nextDayShuffled']] * np.stack([perAnimal.nNeurons,
                                                                                     perAnimal.nNeurons],
                                                                                    axis=1)
        shuffleScore = shuffleScore.sum(axis=0) / perAnimal.nNeurons.sum()

        plt.sca(layout.axes["decodingAccrossDays_{}_{}".format(label, i+1)]["axis"])

        for r in perAnimal.itertuples():
            plt.plot([0,1], [r.sameDayScore, r.nextDayScore], lw=style.lw()*r.nNeurons/400.0,
                     c=style.getColor(r.genotype), alpha=0.2)
        for r in perGenotype.itertuples():
            gt = r.Index
            animalsWithGt = perAnimal.query("genotype == '{}'".format(gt))
            sameDaySEM = bootstrapSEM(animalsWithGt.sameDayScore, animalsWithGt.nNeurons)
            nextDaySEM = bootstrapSEM(animalsWithGt.nextDayScore, animalsWithGt.nNeurons)
            plt.errorbar([0,1], [r.sameDayScore, r.nextDayScore], [sameDaySEM, nextDaySEM],
                         c=style.getColor(gt))

        plt.plot([0,1], [shuffleScore.sameDayShuffled, shuffleScore.nextDayShuffled],
                 c=style.getColor("shuffled"))

        plt.ylim(0.45,1)
        plt.xlim(-0.25, 1.25)
        xlab = ("1-3 days\nlater", "4-14 days\nlater", "14+ days\nlater")
        plt.xticks((0,1), ("Same\nday", xlab[i]))
        if i==0:
            plt.yticks(np.linspace(0.5,1,6), np.linspace(50,100,6,dtype=np.int64))
            plt.ylabel("Decoding accuracy (%)")
        else:
            plt.yticks(np.linspace(0.5,1,6), [""]*5)
        sns.despine(ax=plt.gca())


#%%
acc = P.loc[P.label.str.contains('r\.$|o!$')].copy() # only use win-stay, lose-switch trials
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
    
    ax.axhline(0.5, lw=mpl.rcParams['axes.linewidth'], c='k', alpha=.5, ls=':', clip_on=False)
    
    ax.set_ylim((.5,1))
    ax.set_xlim((-.35,2.35))
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
                   mlines.Line2D([0], [0], marker='s', color='k', label='action duration',
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
coefficients_shuffle = (C.query('shuffled == True')
                         .set_index(["genotype","animal","date","neuron","action"])
                         .coefficient)

#%%
for genotype in ("oprm1", "d1", "a2a"):
    ax = layout.axes['{}_corr_m'.format(genotype)]['axis']
    
    coef_grouped = (coefficients.loc[genotype].unstack()[order]
                                .groupby(['animal','date']))
    corr = coef_grouped.corr().unstack()   
    corr_shuffle = (coefficients_shuffle.loc[genotype].unstack()[order]
                                        .groupby(['animal','date'])
                                        .corr().unstack())
    
    pvalues = ttest_1samp(corr - corr_shuffle, 0).pvalue.reshape(6,6)
    sign_matrix = np.array(([''] * 36)).reshape(6,6)
    sign_matrix[pvalues < 0.01] = '*'
    
    weights = coef_grouped.size()
    corr = np.average(corr, axis=0, weights=weights).reshape(6,6)

    corr[np.triu_indices_from(corr)] = np.nan
    corr = np.ma.masked_where(np.isnan(corr), corr)
    sign_matrix = np.ma.masked_where(np.isnan(corr), sign_matrix)

    cm = ax.pcolormesh(corr, cmap=cmocean.cm.balance, vmin=-.5, vmax=.5,
                       edgecolors='none', lw=0)
    for y,x in zip(*np.tril_indices_from(sign_matrix, -1)):
        ax.text(x+.5,y+.22, sign_matrix[(y, x)], ha='center', va='center',
                color='k', fontsize=7)
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
                  ticks=(-.5,.5))
cax.tick_params(axis='x', which='both',length=0)
cb.outline.set_visible(False)


#%%
prob_value_df = (P.set_index(['shuffled','genotype','animal','date','label','actionNo'])
                  .loc[False, ['action','o!','r.','noNeurons']])
prob_value_df['value'] = (actionValues.set_index(['genotype','animal','date','label','actionNo'])
                                      .value)
prob_value_df = prob_value_df.reset_index()
prob_value_df['stay'] = prob_value_df.label.str.endswith('.').astype('int')

    
#%%
data = prob_value_df.query('action in ["pC2L","pC2R"]').dropna().copy()
data = data.loc[data.label.str.contains('o!$|o\.$|r\.$')]

for (gt,label), gdata in data.groupby(['genotype','action']):
    ax = layout.axes['{}_value_ost'.format(gt)]['axis']
    
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
                    lw=0, marker={'R':'>', 'L':'<'}[label[-1]],
                    markersize=2.8, clip_on=False, barsabove=False,
                    alpha=1, markeredgewidth=0, elinewidth=.5)
        ax.fill_between(value_wAvg, stsw_wAvg-stsw_wSem, stsw_wAvg+stsw_wSem,
                        lw=0, alpha=.35, zorder=-1, color=style.getColor(tt))
    
    ax.axhline(.5, ls=':', c='k', alpha=.35, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
    ax.axvline(0, ls=':', c='k', alpha=.35, zorder=-1, lw=mpl.rcParams['axes.linewidth'])
    
    ax.set_ylim((0,1))
    ax.set_xlim((-5,5))
    ax.set_xticks((-5,0,5))
    ax.invert_xaxis()
    if gt == 'a2a':
        ax.set_xlabel('action value')
    ax.set_yticks((0,.5,1))
    if gt == 'd1':
        ax.set_yticklabels((0,50,100))
        ax.set_ylabel('SVM prediction\nP(win-stay)')
    else:
        ax.set_yticklabels(())
    ax.yaxis.set_minor_locator(MultipleLocator(.25))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    sns.despine(ax=ax)


for gt, gdata in data.groupby('genotype'):
    axkde = layout.axes['{}_value_kde'.format(gt)]['axis']

    gdata = gdata.copy()
    gdata['tt'] = gdata.label.str.slice(-2)
    gdata = gdata.set_index(['animal','date'])
    
    bins = np.arange(-5.5, 5.6, .5)
    labels = (np.arange(-5.5, 5.6, .5) +.25)[:-1]
    gdata['bin'] = pd.cut(gdata.value, bins=bins, labels=labels).astype('float')
    gdist = gdata.groupby(['animal','date','tt','bin']).size().reset_index(['tt','bin'])
    gdist = gdist.rename(columns={0:'pct'})
    gdist['pct'] /= gdata.groupby(['animal','date']).size()
    gdist['noNeurons'] = gdata.groupby(['animal','date']).noNeurons.first()
    
    gdist_stats = gdist.groupby(['tt','bin']).pct.agg(['mean','sem']).reset_index('bin')
        
    for tt, ttdata in gdist_stats.groupby('tt'):
        axkde.plot(ttdata.bin, ttdata['mean'], color=style.getColor(tt),
                   lw=.5, clip_on=False)
        axkde.fill_between(ttdata.bin,
                           ttdata['mean']-ttdata['sem'], ttdata['mean']+ttdata['sem'],
                           color=style.getColor(tt), alpha=.35, lw=0,
                           clip_on=False)
    
    axkde.axvline(0, ls=':', c='k', alpha=.35, zorder=-1,
                  lw=mpl.rcParams['axes.linewidth'])
    
    axkde.set_xlim((-5,5))
    axkde.set_xticks(())
    axkde.set_ylim((0,.1))
    axkde.set_yticks((0,.05,.1))
    axkde.set_yticklabels(())
    if gt == 'd1':
        axkde.set_ylabel('% trials')
        axkde.set_yticklabels((0,5,10))
    sns.despine(bottom=True, trim=True, ax=axkde)
    axkde.set_title(gt)


axt = layout.axes['value_ost_legend']['axis']
legend_elements = [mlines.Line2D([0], [0], marker='<', color='k', markersize=2.8,
                                 markeredgewidth=0, label='(left) choice', lw=0),
                   mpatches.Patch(color=style.getColor('r.'), alpha=1,
                                  label='win-stay'),                          
                   mpatches.Patch(color=style.getColor('o.'), alpha=1,
                                  label='lose-stay'),
                   mpatches.Patch(color=style.getColor('o!'), alpha=1,
                                  label='lose-switch'),
                  ]
axt.legend(handles=legend_elements, ncol=len(legend_elements), loc='center',
           mode='expand')
axt.axis('off')


##%%
def getCorr(ttdata):
    corr = ttdata.groupby(['genotype','animal','date'])[['r.','absValue']].corr()
    corr = pd.DataFrame(pd.Series(corr.unstack(-1)[('absValue','r.')],
                                  name='correlation'))
    corr['noNeurons'] = ttdata.groupby(['genotype','animal','date']).noNeurons.first()
    return corr
    
valueProbCorrs = pd.DataFrame()
for tt, ttdata in data.groupby(data.label.str.slice(-2)):
    ttdata = ttdata.copy()
    ttdata['absValue'] = ttdata.value.abs()
    
    corr = getCorr(ttdata)
    
    ttdata['absValue'] = np.random.permutation(ttdata.absValue)        
    r_corr = getCorr(ttdata)

    corr['rand_correlation'] = r_corr['correlation']
    corr['trialType'] = tt
    corr = corr.set_index('trialType', append=True)
    
    valueProbCorrs = valueProbCorrs.append(corr)


for (gt,tt), cs in (valueProbCorrs.query('trialType in ["o.","r."]')
                                  .groupby(['genotype','trialType'])):
    ax = layout.axes['{}_{}_corr'.format(gt,tt)]['axis']
    
    wAvg = analysisStaySwitchDecoding.wAvg(cs, 'correlation', 'noNeurons')
    wSem = analysisStaySwitchDecoding.bootstrap(cs, 'correlation', 'noNeurons')
    r_wAvg = analysisStaySwitchDecoding.wAvg(cs, 'rand_correlation', 'noNeurons')
    r_wSem = analysisStaySwitchDecoding.bootstrap(cs, 'rand_correlation', 'noNeurons')
    
#    ax.bar([0,1], [wAvg, r_wAvg], yerr=[wSem, r_wSem],
#           color=[style.getColor(tt), style.getColor('shuffled')],
#           lw=0, alpha=.5, zorder=1, width=.5)
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
        if gt == 'a2a':
            ax.set_ylabel('|action value| X P(win-stay)\ncorrelation')
            ax.set_yticklabels((.0,.5))
        sns.despine(ax=ax, bottom=True, trim=True)
    else:
        ax.set_axis_off()
    

#%% plot coefficients for pooled data
ax = layout.axes['log_reg_coef']['axis']

coefficients_mean = logRegCoef.mean(axis=0)
coefficients_sem = logRegCoef.sem(axis=0)
trials_back=7

ax.errorbar(-np.arange(1,trials_back+1),
            coefficients_mean[['Y{}'.format(j) for j in range(1,trials_back+1)]].values, 
            yerr=coefficients_sem[['Y{}'.format(j) for j in range(1,trials_back+1)]].values,
            marker='.', markersize=2.5, color=style.getColor('stay'), label='reward',
            clip_on=False)
ax.errorbar(-np.arange(1,trials_back+1),
            coefficients_mean[['N{}'.format(j) for j in range(1,trials_back+1)]].values, 
            yerr=coefficients_sem[['N{}'.format(j) for j in range(1,trials_back+1)]].values,
            marker='.', markersize=2.5, color=style.getColor('switch'), label='no reward',
            clip_on=False)
ax.errorbar(0, coefficients_mean['intercept'], coefficients_sem['intercept'],
            marker='.', markersize=2.5, color='k', label='bias', clip_on=False)

for (gt,a), coefficients in logRegCoef.groupby(['genotype','animal']):
    ax.plot(-np.arange(1,trials_back+1),
            coefficients[['Y{}'.format(j) for j in range(1,trials_back+1)]].values.flatten(),
            color=style.getColor('stay'), label='', alpha=.2, lw=mpl.rcParams['axes.linewidth'],
            clip_on=False)
    ax.plot(-np.arange(1,trials_back+1),
            coefficients[['N{}'.format(j) for j in range(1,trials_back+1)]].values.flatten(),
            color=style.getColor('switch'), label='', alpha=.2, lw=mpl.rcParams['axes.linewidth'],
            clip_on=False)
    ax.scatter(0, coefficients['intercept'], color='k', marker='.',
               s=8, label='', edgecolors='none', alpha=.2, clip_on=False)

ax.axhline(0, zorder=-99, ls=':', c='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])

ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_xticks((0,-7))
ax.set_xlim((-7.35,0))
ax.set_yticks([0, 1, 2, 3])
ax.set_ylim((-.5, 3))
ax.legend(bbox_to_anchor=(.05,1.05), loc='upper left')
ax.set_xlabel('trials back')
ax.set_ylabel('coefficient')
sns.despine(ax=ax)


#%%
def get_logit(df, dep='leftIn'):
    logit = analysisStaySwitchDecoding.sm.Logit(df[dep], df[['value']])
    result = logit.fit(use_t=True, disp=False)
    stats = pd.DataFrame({'t':result.tvalues, 'p':result.pvalues, 'b':result.params})
    
    x = np.arange(-5, 5, .01)
    y = result.predict(np.array([x]).T)
    prediction = pd.DataFrame({'x':x, 'y':y})
    
    return(result, stats, prediction)
    

ax = layout.axes['log_reg_reg']['axis']
df = logRegDF.copy()

ax.set_xlim((-5, 5.6))
ax.set_xticks((-5,0,5))
ax.set_ylim((-.0475, 1))
ax.set_yticks((0,.5,1))
ax.set_yticklabels((0,50,100))
ax.xaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))

# plot lines for pooled data
result, stats, prediction = get_logit(df)
pp = ax.plot(prediction.x, prediction.y, c='darkgray',
             alpha=1, zorder=1, clip_on=False)

## plot binned scatter data
bins = np.arange(-5.5,5.6)
df['bins'] = pd.cut(df.value, bins=bins)
scatter_means = df.groupby('bins')[['value','leftIn']].mean()
scatter_sems = df.groupby('bins')[['value','leftIn']].sem()
eb = ax.errorbar(scatter_means['value'], scatter_means['leftIn'],
                 yerr=scatter_sems['leftIn'], xerr=scatter_sems['value'],
                 fmt='.', c='darkgray', zorder=2, marker='.', 
                 markersize=2.5, clip_on=False)

stay_hist = (df.groupby(['animal','bins'])[['value','switch']].mean()
               .groupby('bins').agg(['mean','sem']))
hist = ax.bar(bins[:-1]+.5, stay_hist.switch['mean'],
              yerr=stay_hist.switch['sem'], color=style.getColor('switch'),
              alpha=.5, lw=0, width=.8, clip_on=False, zorder=99)

ax.legend(handles=(eb, hist), labels=('left\nchoice','switch'),
          bbox_to_anchor=(.6,1.16), loc='upper left')
ax.axvline(0, zorder=-99, ls=':', c='k', alpha=.35,
           lw=mpl.rcParams['axes.linewidth'])
ax.axhline(.5, zorder=-99, ls=':', c='k', alpha=.35,
           lw=mpl.rcParams['axes.linewidth'])
ax.set_ylabel('% trials')
ax.set_xlabel('action value')

ax.invert_xaxis()
sns.despine(ax=ax)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecoding.svg")


#%%###############################################################################
#fig, axs = plt.subplots(2,1)
#analysisStaySwitchDecoding.drawCoefficientWeightedAverage(endoDataPath, C, 'oprm1', 'pC2L',
#                                                          axs)
#plt.show()
#
#%%
for (gt,ta), gdata in (crossDecoding.query('(trainAction == "pC2L" & testAction == "mC2L") | '+
                                           '(trainAction == "pC2R" & testAction == "mC2R")')
                                    .groupby(['genotype','testAction'])):
    ax = layout.axes['{}_{}_cross'.format(gt,ta)]['axis']
    
    wAvg = analysisStaySwitchDecoding.wAvg(gdata, 'accuracy', 'noNeurons')
    wSem = analysisStaySwitchDecoding.bootstrap(gdata, 'accuracy', 'noNeurons')
    s_wAvg = analysisStaySwitchDecoding.wAvg(gdata, 'accuracy_shuffle', 'noNeurons')
    s_wSem = analysisStaySwitchDecoding.bootstrap(gdata, 'accuracy_shuffle', 'noNeurons')
    
    ax.errorbar(0, wAvg, yerr=wSem, color=style.getColor(ta), clip_on=False,
                marker='v', markersize=3.6, markerfacecolor='w',
                markeredgewidth=.8)
    ax.errorbar(1, s_wAvg, yerr=s_wSem, color=style.getColor(ta), clip_on=False,
                marker='o', markersize=3.2, markerfacecolor='w',
                markeredgewidth=.8)
    ax.plot([0,1], [wAvg, s_wAvg], color=style.getColor(ta), clip_on=False)
    
    for acc in gdata[['accuracy','accuracy_shuffle','noNeurons']].values:
        ax.plot([0,1], acc[:2], lw=mpl.rcParams['axes.linewidth'], alpha=.2,
                clip_on=False, zorder=-99, color=style.getColor(ta))
    
    ax.axhline(.5, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])

    ax.set_ylim((.5,1))    
    ax.set_xlim((-.35,1.35))
    if ta == 'mC2L':
        ax.set_xticks(())
        ax.set_yticks((.5,.75,1.))
        ax.set_yticklabels(())
        if gt == 'a2a':
            ax.set_ylabel('decoding accuracy (%)')
            ax.set_yticklabels((50,75,100))
        sns.despine(ax=ax, bottom=True, trim=True)
    else:
        ax.set_axis_off()

#%%
real, shuffle = analysisStaySwitchDecoding.crossDecodeStaySwitch(endoDataPath)

#%%
cross_df = real.copy()
#%%
cross_df = cross_df.set_index(['genotype','animal','date','trainAction','testAction'])
cross_df = (cross_df.groupby(['genotype','trainAction','testAction'])
                    .apply(analysisStaySwitchDecoding.wAvg, 'accuracy','noNeurons'))
#%%
order = ['mL2C','mC2L','pC2L','pC2R','mC2R','mR2C']
for gt, df in cross_df.groupby('genotype'):
    df = df.unstack().loc[gt].reindex(order)[order]

    plt.figure(figsize=(2.25,2))
#    plt.pcolormesh(df, vmin=0, vmax=1, cmap=cmocean.cm.balance,
#                   edgecolors='none')
    sns.heatmap(df, vmin=0, vmax=1, cmap=cmocean.cm.balance, square=True, 
                cbar=True, annot=df, fmt='.02f')
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.yticks(np.arange(.5, 5.6), order)
    plt.xticks(np.arange(.5, 5.6), order)
    #plt.colorbar()
    #plt.gca().set_aspect('equal')
    plt.suptitle(gt, y=.95)
    plt.show()