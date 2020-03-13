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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
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
layout = figurefirst.FigureLayout(templateFolder / "staySwitchDecodingSupp1.svg")
layout.make_mplfigures()


#%%
palette = {gt: style.getColor(gt) for gt in ['d1','a2a','oprm1']}

# plot histograms
for a, adata in staySwitchAUC.groupby('action'):
    # TODO: fix this
    if a.startswith('d') or a.startswith('pL2C') or a.startswith('pR2C'): continue
    ax = layout.axes['{}_auc_kde'.format(a)]['axis']
    
    for gt, agdata in adata.groupby('genotype'):
#        ax.hist(agdata['auc'], bins=np.arange(-1,1.1,.1), histtype='step',
#                color=style.getColor(gt), label=gt, density=True,
#                lw=2, alpha=.8)
        sns.distplot(agdata['auc'], bins=np.arange(-1,1.1,.1),
                     ax=ax, color=style.getColor(gt), hist=False,
                     kde_kws={'clip_on':False, 'alpha':.75})
    
    ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
    ax.set_ylim((0,2))
    ax.set_yticks((0,1,2))
    ax.set_yticklabels(())
    ax.set_ylabel('')
    if a == 'mL2C':
        ax.set_ylabel('density')
        ax.set_yticklabels(ax.get_yticks())
    ax.set_xlim((-1,1))
    ax.set_xticks(())
    ax.set_xlabel('')
    sns.despine(bottom=True, trim=True, ax=ax)
    
    ax = layout.axes['{}_auc_bp'.format(a)]['axis']

    sns.boxplot('auc', 'genotype', data=adata, ax=ax, 
                palette=palette, saturation=.85, showcaps=False, showfliers=False,
                boxprops={'alpha':0.75, 'lw':0, 'zorder':-99, 'clip_on':False}, 
                width=.75, whiskerprops={'c':'k','zorder':99, 'clip_on':False},
                medianprops={'c':'k','zorder':99, 'clip_on':False})
    
    ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
    ax.set_xlim((-1,1))
    ax.set_ylim((-.75,2.75))
    ax.set_xticks((-1,0,1))
    ax.set_xticks((-.5,.5), minor=True)
    ax.set_xlabel('')
    ax.set_yticks(())
    ax.set_ylabel('')
    sns.despine(left=True, trim=True, ax=ax)
    
axt = layout.axes['auc_legend']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.75,
                                 label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1'}[gt])
                   for gt in ['d1','a2a','oprm1']
                  ]
axt.legend(handles=legend_elements, ncol=3, loc='center',
           mode='expand')
axt.axis('off')


#%%
staySwitchAUC['sign'] = (staySwitchAUC.pct > .995).astype('int') - (staySwitchAUC.pct < .005).astype('int')
cmap = (mpl.colors.LinearSegmentedColormap
                  .from_list('cmap', [sns.color_palette('bright')[4],
                                      (.9,.9,.9),
                                      sns.color_palette('bright')[2]]))

for genotype in ['d1','a2a','oprm1']:   
    df = staySwitchAUC.query('genotype == @genotype & sign in [1,-1]').copy()
    
    for action, pop_df in df.groupby('action'):
        if action.startswith('d') or action.startswith('pL2C') or action.startswith('pR2C'): continue
        axs = [layout.axes['{}_{}_avg'.format(genotype,action+tt)]['axis']
                   for tt in ['r.','o.','o!']]
        cax = layout.axes['colorbar']['axis']
        
        analysisStaySwitchDecoding.drawPopAverageFV('data/endoData_2019.hdf', pop_df, axs, cax,
                                                    auc_weigh=True, saturation=.2,
                                                    smoothing=5.5, cmap=cmap)

#qcax.text(0, 1.01, 'tuning-weighted\nz-score', ha='center', va='bottom', fontdict={'fontsize':6})


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecodingSupp1.svg")


#%%
layout = figurefirst.FigureLayout(templateFolder / "staySwitchDecodingSupp2.svg")
layout.make_mplfigures()


#%%
df = P.loc[P.label.str.contains('r\.$|o!$') & (P.shuffled == False)].copy()

for action, acc_df in df.groupby('action'):
    if action.startswith('d') or action.startswith('pL2C') or action.startswith('pR2C'): continue
    acc_df = acc_df.copy()
    acc_df['trialType'] = acc_df.label.str.slice(-2)
    acc_groupby = acc_df.groupby(['duration','genotype','animal','date','noNeurons'])
    
    n_accuracy = acc_groupby.apply(lambda s: np.mean(s.prediction == s.label))
    d_accuracy = acc_groupby.apply(lambda s: np.mean(s.duration_prediction == s.label))
    
    accuracy = pd.concat([n_accuracy, d_accuracy], keys=['activity','speed'],
                         axis=1).reset_index(['noNeurons','duration'])

    ax = layout.axes['{}_acc_v_speed'.format(action)]['axis']
    axbp = layout.axes['{}_acc_v_speed_bp'.format(action)]['axis']
    
    for gt, gdata in accuracy.groupby('genotype'):   
        means = gdata.groupby('duration')[['activity','speed']].agg(['mean','sem'])
        
        ax.fill_between(means.index.values,
                        (means.activity['mean']-means.activity['sem']).values,
                        (means.activity['mean']+means.activity['sem']).values,
                        lw=0, alpha=.2, color=style.getColor(gt))
        ax.plot(means.index.values, means.activity['mean'].values,
                'v-', markersize=3.6, color=style.getColor(gt), zorder=99, alpha=.75,
                markerfacecolor='w', markeredgewidth=mpl.rcParams['lines.linewidth'])
        
        ax.fill_between(means.index.values,
                        (means.speed['mean']-means.speed['sem']).values,
                        (means.speed['mean']+means.speed['sem']).values,
                        lw=0, alpha=.2, color=sns.desaturate(style.getColor(gt),1))
        ax.plot(means.index.values, means.speed['mean'].values,
                's-', markersize=2.8, color=sns.desaturate(style.getColor(gt),1), zorder=99,
                alpha=.75, markerfacecolor='w', markeredgewidth=mpl.rcParams['lines.linewidth'])
        
        ax.axhline(.5, ls=':', alpha=.5, color='k', lw=mpl.rcParams['axes.linewidth'])
    
#    frac_wst = (acc_df.groupby(['genotype','animal','date','duration','trialType'])
#                      .size().unstack('trialType').fillna(0))
#    frac_wst['frac'] = frac_wst['r.'] / (frac_wst['r.'] + frac_wst['o!'])
#    frac_wst = frac_wst.groupby('duration').frac.agg(['mean','sem'])
#    
#    ax.errorbar(frac_wst.index, frac_wst['mean'], frac_wst['sem'],
#                color='k', zorder=1000, alpha=.8, ls='--')
    
    sns.boxplot('duration', 'trialType', data=acc_df, order=['r.','o!'],
                saturation=.85, showcaps=False,  showfliers=False,
                palette={tt:style.getColor(tt) for tt in ['r.','o!']},
                boxprops={'alpha':.8, 'lw':0, 'zorder':-99, 'clip_on':False}, width=.65, 
                whiskerprops={'c':'k','zorder':99,'clip_on':False},
                capprops={'clip_on':False}, medianprops={'c':'k','zorder':99},
                ax=axbp)
    
    ax.set_ylim((0,1))
    #ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.set_yticks((0,.25,.5,.75,1))
    ax.set_yticklabels(())
    if action == 'mL2C':
        ax.set_yticklabels((ax.get_yticks()*100).astype('int'))
        ax.set_ylabel('decoder accuracy (%)')
        #ax.set_xlabel('movement duration (s)')
    ax.xaxis.set_major_locator(MultipleLocator(.1))
    ax.xaxis.set_minor_locator(MultipleLocator(.05))
#    pctls = (np.concatenate(acc_df.groupby('label').duration
#                                  .apply(np.percentile, [25,75]).tolist()))
#    xlims = (pctls.min()-.025, pctls.max()+.025)
    xlims = {'mL2C':(.325,.775), 'mR2C':(.325,.775),
             'pC2L':(.225,.575), 'pC2R':(.225,.575),
             'mC2L':(.275,.525), 'mC2R':(.275,.525)}[action]
    ax.set_xlim(xlims)
    sns.despine(ax=ax)
    
    axbp.vlines(xlims, -.5, 1.5, ls=':', color='k', alpha=.5,
                lw=mpl.rcParams['axes.linewidth'])
    axbp.set_ylabel('')
    axbp.set_xlabel('')
    axbp.set_yticks(())
    axbp.set_xlim((0,1))
    axbp.xaxis.set_major_locator(MultipleLocator(.5))
    axbp.xaxis.set_minor_locator(MultipleLocator(.05))
    sns.despine(left=True, bottom=True, top=False, ax=axbp)
    axbp.xaxis.tick_top()


axt = layout.axes['acc_v_speed_legend1']['axis']
legend_elements = [mlines.Line2D([0], [0], marker='v', color='k', label='neural activity',
                                 markerfacecolor='w', markersize=3.6,
                                 markeredgewidth=.75),
                   mlines.Line2D([0], [0], marker='s', color='k', label='action duration',
                                 markerfacecolor='w', markersize=2.8,
                                 markeredgewidth=.75)
                  ]
axt.legend(handles=legend_elements, loc='center', ncol=2, mode='expand')
axt.axis('off')

axt = layout.axes['acc_v_speed_legend2']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.75,
                                  label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1'}[gt])
                   for gt in ['d1','a2a','oprm1']]
axt.legend(handles=legend_elements, loc='center', ncol=3, mode='expand')
axt.axis('off')    

axbpt = layout.axes['acc_v_speed_bp_legend']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(tt), alpha=.8,
                                 label={'r.':'win-stay','o!':'lose-switch'}[tt])
                   for tt in ['r.','o!']
                  ]
axbpt.legend(handles=legend_elements, ncol=2, loc='center')
axbpt.axis('off')


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

for label in ('mL2C','pC2L','mC2L','mR2C','pC2R','mC2R'):
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

        plt.sca(layout.axes["decodingAcrossDays_{}_{}".format(label, i+1)]["axis"])

        for r in perAnimal.itertuples():
            plt.plot([0,1], [r.sameDayScore, r.nextDayScore],
                     lw=mpl.rcParams['axes.linewidth']*r.nNeurons/400.0, alpha=.25,
                     clip_on=False, zorder=-99, color=style.getColor(r.genotype))
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
        plt.xlabel(("1-3", "4-13", "14+")[i] + "\ndays", labelpad=7, fontsize=6)
        if i==0:
            plt.yticks(np.linspace(.5,1,3), np.linspace(50,100,3,dtype=np.int64))
            plt.ylabel("decoding accuracy (%)")
        else:
            plt.yticks([])
        plt.axhline(0.5, color='k', lw=0.5, alpha=0.5, ls=":")
        sns.despine(ax=plt.gca(), left=(i!=0), bottom=True)

axt = layout.axes['across_days_legend1']['axis']
legend_elements = [mpl.lines.Line2D([0],[0], color=style.getColor(gt),
                                    label={'oprm1':'Oprm1','a2a':'A2A','d1':'D1',
                                           'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']]
axt.legend(handles=legend_elements, loc='center', ncol=4, mode='expand')
axt.axis('off')

#%%
def getCorr(data):
    corr = data.groupby(['genotype','animal','date'])[['r.','value']].corr()
    corr = pd.DataFrame(pd.Series(corr.unstack(-1)[('value','r.')],
                                  name='correlation'))
    corr['noNeurons'] = data.groupby(['genotype','animal','date']).noNeurons.first()
    return corr

def randomShiftValue(data):
    def shift(v):
        v = pd.Series(np.roll(v, np.random.randint(10,30) * np.random.choice([-1,1])),
                      index=v.index)
        return v
    
    data = data.copy()
    data['value'] = data.groupby(['genotype','animal','date'])['value'].apply(shift).copy()
    return data


prob_value_df = (P.set_index(['shuffled','genotype','animal','date','label','actionNo'])
                  .loc[False, ['action','o!','r.','noNeurons']])
prob_value_df['value'] = (actionValues.set_index(['genotype','animal','date','label','actionNo'])
                                      .value)
prob_value_df = prob_value_df.reset_index()
prob_value_df['stay'] = prob_value_df.label.str.endswith('.').astype('int')

  
##%%
for p, actions in enumerate((['mL2C','mR2C'],['mC2L','mC2R'])):
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
        ax.invert_xaxis()
        if gt == 'a2a':
            ax.set_xlabel('action value')
        ax.set_yticks((0,.5,1))
        if (gt == 'd1') & (p == 0):
            ax.set_yticklabels((0,50,100))
            ax.set_ylabel('SVM prediction\nP(win-stay)')
        else:
            ax.set_yticklabels(())
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))
        sns.despine(ax=ax)
    
    
    for gt, gdata in data.groupby('genotype'):
        axkde = layout.axes['{}_value_kde_{}'.format(gt,p+1)]['axis']
    
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
        if (gt == 'd1') & (p == 0):
            axkde.set_ylabel('% trials')
            axkde.set_yticklabels((0,5,10))
        sns.despine(bottom=True, trim=True, ax=axkde)
        axkde.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[gt])
    
    
    axt = layout.axes['value_ost_legend_{}'.format(p+1)]['axis']
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
for actions in (['mL2C','mR2C'],['pC2L','pC2R'],['mC2L','mC2R']):
    data = prob_value_df.query('action in @actions').dropna().copy()
    data = data.loc[data.label.str.contains('o!$|o\.$|r\.$')]
    
    valueProbCorrs = pd.DataFrame()
    for label, ldata in data.groupby(data.label):
        ldata = ldata.copy()
        if 'R' in label:
            ldata['value'] = ldata.value * -1
        #ldata['absValue'] = ldata.value.abs()
        
        corr = getCorr(ldata)
        
        #ldata['value'] = np.random.permutation(ldata.value)  
        ldata_vshifted = randomShiftValue(ldata)
        r_corr = getCorr(ldata_vshifted)
    
        corr['rand_correlation'] = r_corr['correlation']
        corr['label'] = label
        corr = corr.set_index('label', append=True)
        
        valueProbCorrs = valueProbCorrs.append(corr)
    
    
    for (gt,label), cs in (valueProbCorrs.groupby(['genotype','label'])):
        ax = layout.axes['{}_{}_corr'.format(gt,label)]['axis']
        
        wAvg = analysisStaySwitchDecoding.wAvg(cs, 'correlation', 'noNeurons')
        wSem = analysisStaySwitchDecoding.bootstrap(cs, 'correlation', 'noNeurons')
        r_wAvg = analysisStaySwitchDecoding.wAvg(cs, 'rand_correlation', 'noNeurons')
        r_wSem = analysisStaySwitchDecoding.bootstrap(cs, 'rand_correlation', 'noNeurons')
        
        ax.errorbar(0, wAvg, yerr=wSem, color=style.getColor(label[-2:]), clip_on=False,
                    marker='v', markersize=3.6, markerfacecolor='w',
                    markeredgewidth=.8)
        ax.errorbar(1, r_wAvg, yerr=r_wSem, color=style.getColor(label[-2:]), clip_on=False,
                    marker='o', markersize=3.2, markerfacecolor='w',
                    markeredgewidth=.8)
        ax.plot([0,1], [wAvg, r_wAvg], color=style.getColor(label[-2:]), clip_on=False)
        
        for c in cs[['correlation','rand_correlation','noNeurons']].values:
            ax.plot([0,1], c[:2], lw=mpl.rcParams['axes.linewidth'], alpha=.2,
                    clip_on=False, zorder=-99, color=style.getColor(label[-2:]))
        
        ax.axhline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
    
        ax.set_ylim((0,.5))    
        ax.set_xlim((-.35,1.35))
        if label[-2:] == 'r.':
            ax.set_xticks(())
            ax.set_yticks((0,.5))
            ax.set_yticklabels(())
            ax.set_yticks((.25,), minor=True)
            if (label == "mL2Cr.") & (gt == 'a2a'):
                ax.set_ylabel('action value X P(win-stay)\ncorrelation')
                ax.set_yticklabels((.0,.5))
            sns.despine(ax=ax, bottom=True, trim=True)
        else:
            ax.set_axis_off()
    

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecodingSupp2.svg")
