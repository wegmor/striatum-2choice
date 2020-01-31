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
from matplotlib.ticker import MultipleLocator, FixedLocator
import pathlib
import figurefirst
import style
import analysisStaySwitchDecoding
import cmocean
#from scipy.stats import ttest_1samp
import scipy.stats
#import statsmodels.formula.api as smf
from scipy.spatial.distance import pdist, squareform
from utils import readSessions, fancyViz, sessionBarPlot
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
    
cachedDataPath = cacheFolder / 'actionValues_shuffled.pkl'
if cachedDataPath.is_file():
    actionValues_shuffled = pd.read_pickle(cachedDataPath)
else:
    actionValues_shuffled = analysisStaySwitchDecoding.getActionValues(endoDataPath, logRegCoef,
                                                                       on_shuffled=True)
    actionValues_shuffled.to_pickle(cachedDataPath)
    
#cachedDataPath = cacheFolder / "staySwitchCrossDecoding.pkl"
#if cachedDataPath.is_file():
#    crossDecoding = pd.read_pickle(cachedDataPath)
#else:
#    crossDecoding, shuffleCross = (analysisStaySwitchDecoding
#                                       .crossDecodeStaySwitch(endoDataPath))
#    crossDecoding = crossDecoding.set_index(['genotype','animal','date',
#                                             'testAction','trainAction'])
#    shuffleCross = shuffleCross.set_index(['genotype','animal','date',
#                                           'testAction','trainAction'])
#    crossDecoding['accuracy_shuffle'] = shuffleCross.accuracy
#    crossDecoding = (crossDecoding[['noNeurons','accuracy','accuracy_shuffle']]
#                                  .reset_index())
#    crossDecoding.to_pickle(cachedDataPath)
    
cachedDataPath = cacheFolder / 'staySwitchAUC.pkl'
if cachedDataPath.is_file():
    staySwitchAUC = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath,
                                                                  n_shuffles=1000)
    staySwitchAUC.to_pickle(cachedDataPath)
    
cachedDataPath = cacheFolder / 'staySwitchAUC_shuffled.pkl'
if cachedDataPath.is_file():
    staySwitchAUC_shuffled = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC_shuffled = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath,
                                                                           n_shuffles=1,
                                                                           on_shuffled=True)
    staySwitchAUC_shuffled.to_pickle(cachedDataPath)


#%% Example neurons
examples = [("5308", "190131", 292, "oprm1"),
            ("5643", "190114", 178, "d1")]

for p, (a, d, n, gt) in enumerate(examples):
    s = next(readSessions.findSessions(endoDataPath, genotype=gt,
                                       animal=a, date=d, task='2choice'))
    traces = s.readDeconvolvedTraces(rScore=True)
    
    lfa = s.labelFrameActions(reward='fullTrial', switch=True).set_index(traces.index)
    d_labels = ((lfa.set_index('actionNo').label.str.slice(0,5) + \
                 lfa.groupby('actionNo').label.first().shift(1).str.slice(4))
                .reset_index().set_index(lfa.index))
    lfa.loc[lfa.label.str.contains('d.$'), 'label'] = d_labels.fillna('-')
    
    trace = traces[n]
    
    for trialType in ['r.','o.','o!']:
        axfv = layout.axes['f8_ex{}_{}'.format(p+1, trialType)]['axis']
        fv = fancyViz.SchematicIntensityPlot(s, splitReturns=False,
                                             linewidth=mpl.rcParams['axes.linewidth'],
                                             smoothing=7.5, saturation=1.5)
        fv.setMask(lfa.label.str.endswith(trialType).values)
        img = fv.draw(trace, ax=axfv)

cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.55, -.3, '-1.5', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.55, -.3, '1.5', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})


#%% plot stay vs switch ROC AUC tuning distributions
#  could be done with "tuning" defined as z-scored by shuffle dist?
staySwitch = staySwitchAUC.loc[staySwitchAUC.action.isin(['mR2C'])].copy()
staySwitch_shuffled = staySwitchAUC_shuffled[staySwitchAUC_shuffled.action.isin(['mR2C'])].copy()
staySwitch_shuffled['genotype'] = 'shuffled'
staySwitch = pd.concat([staySwitch, staySwitch_shuffled])
palette = {gt: style.getColor(gt) for gt in ['d1','a2a','oprm1','shuffled']}

# plot kde
for a, adata in staySwitch.groupby('action'):
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
    if a == 'mR2C':
        ax.set_ylabel('density')
        ax.set_yticklabels(ax.get_yticks())
    ax.set_xlim((-1,1))
    #ax.set_xticks(())
    ax.set_xticks((-1,0,1))
    ax.set_xticklabels((-1,0,1))
    ax.set_xticks((-.5,.5), minor=True)
    ax.set_xlabel('selectivity score')
    sns.despine(bottom=False, trim=True, ax=ax)
    
#    ax = layout.axes['{}_auc_bp'.format('pL' if 'L' in a else 'pR')]['axis']
#
#    sns.boxplot('auc', 'genotype', data=adata, ax=ax, 
#                palette=palette, saturation=.85, showcaps=False, showfliers=False,
#                boxprops={'alpha':0.75, 'lw':0, 'zorder':-99, 'clip_on':False}, 
#                width=.75, whiskerprops={'c':'k','zorder':99, 'clip_on':False},
#                medianprops={'c':'k','zorder':99, 'clip_on':False},
#                order=['d1','a2a','oprm1','shuffled'])
#    
#    ax.axvline(0, ls=':', color='k', alpha=.5, lw=mpl.rcParams['axes.linewidth'])
#    ax.set_xlim((-1,1))
#    ax.set_ylim((-.75,3.75))
#    ax.set_xticks((-1,0,1))
#    ax.set_xticklabels((-1,'',1))
#    ax.set_xticks((-.5,.5), minor=True)
#    ax.set_xlabel('')
#    ax.set_yticks(())
#    ax.set_ylabel('')
#    sns.despine(left=True, trim=True, ax=ax)
    
axt = layout.axes['auc_legend']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.75,
                                 label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1',
                                        'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']
                  ]
axt.legend(handles=legend_elements, ncol=2, loc='center',
           mode='expand')
axt.axis('off')


#%% session bar plots showing fractions of stay-switch tuned neurons per session
action_aucs = staySwitchAUC.query('action == "mR2C"').copy()
action_aucs['stay'] = action_aucs.pct > .995
action_aucs['switch'] = action_aucs.pct < .005

sign_sess_frac = pd.DataFrame(action_aucs.groupby(['genotype','animal','date'])[['stay']].sum())
sign_sess_frac['switch'] = action_aucs.groupby(['genotype','animal','date']).switch.sum()
sign_sess_frac['noNeurons'] = action_aucs.groupby(['genotype','animal','date']).size()
sign_sess_frac.loc[:,['stay','switch']] = (sign_sess_frac[['stay','switch']] / 
                                           sign_sess_frac.noNeurons.values[:,np.newaxis])
sign_sess_frac.reset_index(inplace=True)

for tuning in ['stay','switch']:
    ax = layout.axes['perc_{}'.format(tuning)]
    sessionBarPlot.sessionBarPlot(sign_sess_frac, tuning, ax, style.getColor,
                                  weightScale=.0075, orientation='vertical')
    ax.axhline(0, ls=':', lw=mpl.rcParams['axes.linewidth'], color='k', alpha=.5,
               clip_on=False)
#    ax.axvline(0, ls=':', lw=mpl.rcParams['axes.linewidth'], color='k', alpha=.5,
#               clip_on=False)
    ax.set_ylim((0,.3))
    ax.set_yticks((0,.3))
    ax.set_yticks((.1,.2), minor=True)
    ax.set_yticklabels(())
    if tuning == "switch":
        ax.set_yticklabels((ax.get_yticks() * 100).astype('int'))
        ax.set_ylabel('% selective')
    ax.set_xticks(())
    ax.set_title('{}'.format({'stay':'win-stay','switch':'lose-switch'}[tuning]),
                 pad=4)
    sns.despine(ax=ax, trim=False, bottom=True)
#    if tuning == 'switch':
#        ax.invert_xaxis()


#%% pie charts
df = staySwitchAUC.copy()
df = df.query('action not in ["pL2C","pR2C"]')
df['sign'] = (df.pct > .995) | (df.pct < .005)

# only keep max tuning for each neuron
maxdf = (df.loc[df.groupby(['genotype','animal','date','neuron'])
                  .tuning.apply(lambda t: t.abs().idxmax())])
# inidcate whether stay or switch tuned
maxdf.loc[maxdf.tuning.apply(np.sign) == 1, 'action'] += 'r.'
maxdf.loc[maxdf.tuning.apply(np.sign) == -1, 'action'] += 'o!'
maxdf.loc[~maxdf.sign, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['sign']].count() # get counts

# create dictionary with modified alpha to separate center port phases
cdict = {a:style.getColor(a[:4]) for a in maxdf.reset_index().action.unique()}
cdict['none'] = np.array((1,1,1))
cdict['pC2Lr.'] = np.append(cdict['pC2Lr.'], .45)
cdict['pC2Lo!'] = np.append(cdict['pC2Lo!'], .45)


for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']

    order = ['mC2Ro!','mC2Rr.','mL2Co!','mL2Cr.',
             'mC2Lo!','mC2Lr.','mR2Co!','mR2Cr.',
             'none',
             'dL2Co!','dL2Cr.','dR2Co!','dR2Cr.',
             'pC2Lo!','pC2Lr.','pC2Ro!','pC2Rr.']
    gdata = maxdf.loc[g].loc[order]
    ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
           explode=[.1]*len(gdata), textprops={'color':'k'},
           colors=[cdict[a] for a in gdata.index])
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'k'},
                    explode=[.1]*len(gdata))
    for (w,a) in zip(ws,order):
        w.set_fill(False)
        if a.endswith('o!'):
            w.set_hatch('X'*10)
    
    ax.set_aspect('equal')

ax = layout.axes['pie_legend']['axis']
legend_elements = [mpatches.Wedge((0,0), 1, 80,100, edgecolor='k', facecolor='w', lw=.35,
                                  label='win-stay'),
                   mpatches.Wedge((0,0), 1, 80, 100, edgecolor='k', facecolor='w', lw=.35,
                                  hatch='X'*10, label='lose-switch')]
ax.legend(handles=legend_elements, loc='center')
ax.axis('off')


##%% tuning counts (simple)
hist_df = analysisStaySwitchDecoding.getTunedNoHistData(df)

axs = {}
for g, gdata in hist_df.query('bin != 0').groupby('genotype'):
    ax = layout.axes['no_tuned_'+g]['axis']
    axs[g] = ax
    
    ax.scatter(analysisStaySwitchDecoding.jitter(gdata.bin, .12), gdata.sign,
               s=gdata.noNeurons/25, edgecolor=style.getColor(g),
               facecolor='none', alpha=.8, zorder=0, clip_on=False,
               lw=mpl.rcParams['axes.linewidth'])
    
    avg = gdata.groupby('bin').apply(analysisStaySwitchDecoding.wAvg, 'sign', 'noNeurons')
    sem = gdata.groupby('bin').apply(analysisStaySwitchDecoding.bootstrap, 'sign', 'noNeurons')
    ax.bar(avg.index, avg, yerr=sem, color=style.getColor(g),
           lw=0, alpha=.3, zorder=1)
    
    ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])
    ax.set_ylim((0,.5))
    ax.set_yticks((0,.25,.5))
    ax.set_yticklabels(())
    ax.yaxis.set_minor_locator(MultipleLocator(.125))
    ax.set_xlim((0.25,5.75))
    ax.set_xticks((1,3,5))
    ax.xaxis.set_minor_locator(FixedLocator((2,4)))
    ax.set_xticklabels(['1','3','5+'])
    sns.despine(ax=ax)
    

axs['d1'].set_yticklabels((0,25,50))
axs['d1'].set_ylabel('neurons (%)')
axs['a2a'].set_xlabel('number of actions')


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
eb = ax.errorbar(0, coefficients_mean['intercept'], coefficients_sem['intercept'],
                 marker='.', markersize=2.5, color='k', label='bias', clip_on=False)
[l.set_clip_on(False) for l in eb[2]]

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
ax.legend(bbox_to_anchor=(.05,1.04), loc='upper left')
ax.set_xlabel('trials back')
ax.set_ylabel('coefficient')
sns.despine(ax=ax)


#%% logistic regression plot
def get_logit(df, dep='rightIn'):
    logit = analysisStaySwitchDecoding.sm.Logit(df[dep], df[['value']])
    result = logit.fit(use_t=True, disp=False)
    stats = pd.DataFrame({'t':result.tvalues, 'p':result.pvalues, 'b':result.params})
    
    x = np.arange(-5, 5, .01)
    y = result.predict(np.array([x]).T)
    prediction = pd.DataFrame({'x':x, 'y':y})
    
    return(result, stats, prediction)
    

ax = layout.axes['log_reg_reg']['axis']
df = logRegDF.copy()

ax.set_xlim((-5.6, 5))
ax.set_xticks((-5,0,5))
ax.set_ylim((-.02, 1))
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
# take bin mean per animal -> mean and sem per bin
scatter_means = df.groupby(['animal','bins'])[['value','rightIn']].mean().groupby('bins').mean()
scatter_sems = df.groupby(['animal','bins'])[['value','rightIn']].mean().groupby('bins').sem()
eb = ax.errorbar(scatter_means['value'], scatter_means['rightIn'],
                 yerr=scatter_sems['rightIn'], xerr=scatter_sems['value'],
                 fmt='.', c='darkgray', zorder=2, marker='.', 
                 markersize=2.5, clip_on=False)

stay_hist = (df.groupby(['animal','bins'])[['value','switch']].mean()
               .groupby('bins').agg(['mean','sem']))
hist = ax.bar(bins[:-1]+.5, stay_hist.switch['mean'],
              yerr=stay_hist.switch['sem'], color=style.getColor('switch'),
              alpha=.5, lw=0, width=.8, clip_on=False, zorder=99)

ax.legend(handles=(eb, hist), labels=('right\nchoice','switch'),
          bbox_to_anchor=(.05,1.04), loc='upper left')
ax.axvline(0, zorder=-99, ls=':', c='k', alpha=.35,
           lw=mpl.rcParams['axes.linewidth'])
ax.axhline(.5, zorder=-99, ls=':', c='k', alpha=.35,
           lw=mpl.rcParams['axes.linewidth'])
ax.set_ylabel('% trials')
ax.set_xlabel('action value')

#ax.invert_xaxis()
sns.despine(ax=ax)
   

#%% example population rasters etc.
action = 'mR2C'
stay_aucs = staySwitchAUC.query('action == @action & pct > .995').copy()
switch_aucs = staySwitchAUC.query('action == @action & pct < .005').copy()
for stsw, aucs in zip(['stay','switch'],[stay_aucs, switch_aucs]):
    stacked = analysisStaySwitchDecoding.getStSwRasterData(endoDataPath, aucs,
                                                           action, pkl_suffix=stsw)
    
    # add whitespace separator between genotypes -- total hack
    sep_rows = stacked.shape[0] // 70
    sep_cols = stacked.shape[1]+4
    sep = pd.DataFrame(np.array([np.nan]*(sep_cols*sep_rows)).reshape((sep_rows,sep_cols)))
    sep.set_index([0,1,2,3], inplace=True)
    sep.index.names = stacked.index.names
    sep.columns = stacked.columns
    stacked = pd.concat([stacked.loc[[gt]].append(sep) for gt in ('d1','a2a')] +
                        [stacked.loc[['oprm1']]])
        
    ccax = layout.axes['{}pop_colorcode'.format(stsw)]['axis']
    pal = [style.getColor(gt) for gt in ['d1','a2a','oprm1']]
    colorcode = (stacked.reset_index().genotype
                        .replace({'d1':0,'a2a':1,'oprm1':2}).values
                        .reshape(len(stacked),1))
    ccax.imshow(colorcode, cmap=mpl.colors.ListedColormap(pal),
                aspect='auto', interpolation='nearest')
    ccax.axis('off')
    
    raxs = [layout.axes['{}pop_raster_{}'.format(stsw, tt)]['axis'] for tt in ('o!','o.','r.')]
    aaxs = [layout.axes['{}pop_avg_{}'.format(stsw, tt)]['axis'] for tt in ('o!','o.','r.')]
    for i, p in enumerate([action+tt for tt in ('o!','o.','r.')]):
        for gt, gdata in stacked[p].groupby('genotype'):
            m = gdata.mean(axis=0)
            sem = gdata.sem(axis=0)
            aaxs[i].fill_between(np.arange(15), m-sem, m+sem, alpha=0.35, lw=0,
                                 color=style.getColor(gt), clip_on=False,
                                 zorder={'d1':3,'a2a':1,'oprm1':2}[gt])
            aaxs[i].plot(m.values, color=style.getColor(gt), clip_on=False,
                         lw=.5, alpha=.75, zorder={'d1':3,'a2a':1,'oprm1':2}[gt])
        aaxs[i].set_ylim(0, .5)
        aaxs[i].set_xlim(-.5, 14.5)
        aaxs[i].hlines([0], -.5, 16, lw=mpl.rcParams['axes.linewidth'],
                       color='k', alpha=.5, zorder=-99, ls=':', clip_on=False)
        aaxs[i].vlines([4.5,9.5], -.18, .5, ls=':', color='k', alpha=1, 
                       lw=mpl.rcParams['axes.linewidth'], clip_on=False,
                       zorder=-99)
        aaxs[i].axis('off')

        img = raxs[i].imshow(stacked[p], aspect="auto", interpolation="nearest",
                             vmin=-.5, vmax=.5, cmap="RdYlBu_r")
        raxs[i].axvline(4.5, ls=':', color='k', alpha=1, 
                       lw=mpl.rcParams['axes.linewidth'])
        raxs[i].axvline(9.5, ls=':', color='k', alpha=1,
                        lw=mpl.rcParams['axes.linewidth'])
        raxs[i].axis('off')
        
    aaxs[0].axis('on')
    sns.despine(ax=aaxs[0], trim=True, left=False, right=True, bottom=True,
                offset=1)
    aaxs[0].set_ylabel("z-score")
    aaxs[0].set_yticks((0,.5))
    aaxs[0].set_yticks((.25,), minor=True)
    #aaxs[2].yaxis.set_label_position('right')
    #aaxs[2].yaxis.set_ticks_position('right')
    aaxs[0].set_xticks(())
    
    raxs[0].axis('on')
    sns.despine(ax=raxs[0], left=True, bottom=True)
#    raxs[2].set_ylabel("neuron", labelpad=-5)
#    raxs[2].set_yticks((0, len(stacked)-1))
#    raxs[2].set_yticklabels([1,len(aucs)])
    ylims = raxs[0].get_ylim()
    gtorder = ['d1','a2a','oprm1']
    yticks = (stacked.groupby('genotype').size().loc[gtorder[:-1]].cumsum().values + 
              [sep_rows,2*sep_rows])
    yticks = (np.array([0] + list(yticks)) +
              stacked.groupby('genotype').size().loc[gtorder].values // 2)
    raxs[0].set_yticks(yticks-1)
    raxs[0].set_yticklabels(['{}\n({})'.format(gt,n) for (n,gt) in 
                                 zip(stacked.groupby('genotype').size().loc[gtorder].values,
                                     ['D1','A2A','Oprm1'])],
                            rotation=0, va='center', ha='center')
    raxs[0].set_ylim(ylims) # somehow not having a 0 tick crops the image!
#    raxs[2].yaxis.set_label_position('right')
#    raxs[2].yaxis.set_ticks_position('right')
    raxs[0].tick_params(axis='y', length=0, pad=13)
    [tick.set_color(style.getColor(gt)) for (tick,gt) in zip(raxs[0].get_yticklabels(),
                                                             gtorder)]
    raxs[0].set_xticks(())
    
#    f8axs = [layout.axes['{}pop_f8_{}'.format(stsw, tt)]['axis'] for tt in ('r.','o.','o!')]
#    [ax.axis('off') for ax in f8axs]
#    analysisStaySwitchDecoding.drawPopAverageFV(endoDataPath, aucs, f8axs,
#                                                saturation=.5)
    
    cax = layout.axes['{}pop_colorbar'.format(stsw)]['axis']
    cb = plt.colorbar(img, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cax.set_axis_off()
    cax.text(-.05, .3, -.5, ha='right', va='center', fontdict={'fontsize':6},
             transform=cax.transAxes)
    cax.text(1.05, .3, .5, ha='left', va='center', fontdict={'fontsize':6},
             transform=cax.transAxes)
    cax.text(0.5, -.1, 'z-score', ha='center', va='top', fontdict={'fontsize':6},
             transform=cax.transAxes)


for stsw, aucs in zip(['stay','switch'],[stay_aucs, switch_aucs]):
    actionMeans = analysisStaySwitchDecoding.getActionMeans(endoDataPath, aucs,
                                                            actionValues, action,
                                                            pkl_suffix=stsw)
    actionMeans.set_index(['genotype','animal','date','neuron'], inplace=True)

    actionMeans['bin'] = pd.cut(actionMeans.value * (1 if 'R' in action else -1),
                                [-10,0,1,2,3,4,10])
    
    # v per neuron bin-average activity and value
    means = actionMeans.groupby(['genotype','animal','neuron','bin'])[['activity','value']].mean()
    # v per genotype
    means = means.groupby(['genotype','bin']).agg(['mean','sem'])
    
    ax = layout.axes['{}pop_av_sd_reg'.format(stsw)]['axis']
    for gt, data in means.groupby('genotype'):
        eb = ax.errorbar(data['value','mean'], data['activity','mean'],
                         yerr=data['activity','sem'], xerr=data['value','sem'],
                         marker='o', markersize=1, color=style.getColor(gt),
                         lw=.5, clip_on=False)
        for lines in eb.lines[2]:
            lines.set_clip_on(False)
        ax.fill_between(data['value','mean'], data['activity','mean']-data['activity','sem'],
                                              data['activity','mean']+data['activity','sem'],
                        color=style.getColor(gt), lw=0, alpha=.35, label=gt, clip_on=False)
        
    ax.axhline(0, ls=':', color='k', alpha=.5, zorder=-99, lw=mpl.rcParams['axes.linewidth'])
    ax.axvline(0, ls=':', color='k', alpha=.5, zorder=-99, lw=mpl.rcParams['axes.linewidth'])
    ax.set_ylim((-.2,.5))
    ax.set_yticks((0,.5))
    ax.set_yticks((.25,), minor=True)
    ax.set_xlim((-1,5) if 'R' in action else (-5,1))
    ax.set_xticks((0,5) if 'R' in action else (-5,0))
    #ax.set_xticks(np.arange(-1,6) if 'L' in action else np.arange(-5,2), minor=True)
    ax.set_xticks((2.5,) if 'R' in action else (-2.5,), minor=True)
    ax.set_ylabel('z-score')
    ax.set_xlabel('action value')
    #ax.legend(loc=(1,.65))
    sns.despine(ax=ax, trim=False)
    
    # correlation plots
    ax = layout.axes['{}pop_corrs'.format(stsw)]['axis']
    
    corrs_df = (actionMeans.groupby(['genotype','animal','date','neuron'])
                           [['activity','duration','value']].corr().unstack())
    mean_corrs = corrs_df.groupby(['genotype','animal','date']).agg(['mean','size'])

    ax.scatter(mean_corrs[('activity','duration','mean')], mean_corrs[('activity','value','mean')],
                          s=mean_corrs[('activity','duration','size')]/5,
                          facecolors=mean_corrs.reset_index().genotype.apply(style.getColor),
                          lw=.35, alpha=.45, clip_on=False)
    
    ax.axhline(0, ls=':', color='k', alpha=.5, zorder=-99, lw=mpl.rcParams['axes.linewidth'])
    ax.axvline(0, ls=':', color='k', alpha=.5, zorder=-99, lw=mpl.rcParams['axes.linewidth'])
    ax.set_aspect('equal')
    ax.set_ylabel('r(activity, value)')
    ax.set_xlabel('r(activity, duration)')
    if stsw == 'switch':
        ax.set_xlim((-.3,.3))
        ax.set_ylim((-.5,.1))
        ax.set_xticks((-.2,0,.2))
        ax.set_yticks((0,-.2,-.4))
        sns.despine(ax=ax, trim=False) # trim=True is buggy -> repeatedly setting ticks
        ax.set_xticks((-.2,0,.2))
        ax.set_yticks((0,-.2,-.4))
    if stsw == 'stay':
        ax.set_xlim((-.3,.3))
        ax.set_ylim((-.1,.5))
        ax.set_xticks((-.2,0,.2))
        ax.set_yticks((0,.2,.4))
        sns.despine(ax=ax, trim=False)
        ax.set_xticks((-.2,0,.2))
        ax.set_yticks((0,.2,.4))


#%% stay-switch decoding accuracy versus shuffled & action duration decoders
acc = P.loc[P.label.str.contains('r\.$|o!$')].copy() # only use win-stay, lose-switch trials
acc_groupby = acc.groupby(['genotype','animal','date','noNeurons','action', 'shuffled'])
acc_activity = acc_groupby.apply(lambda sess: np.mean(sess.prediction == sess.label))
acc_activity.name = 'accuracy'
acc_speed = acc_groupby.apply(lambda sess: np.mean(sess.duration_prediction == sess.label))
acc_speed.name = 'accuracy'
acc = pd.concat([acc_activity, acc_speed], keys=['activity','speed'], names=['decoder'])
acc = acc.reset_index('noNeurons')
acc = acc.reorder_levels((0,5,1,2,3,4))
   
##%%
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
    
#ax = layout.axes['dec_legend']['axis']
#legend_elements = [mlines.Line2D([0], [0], marker='o', color='k', label='neural activity\n(labels shuffled)',
#                                 markerfacecolor='w', markersize=3.2,
#                                 markeredgewidth=.8),
#                   mlines.Line2D([0], [0], marker='v', color='k', label='neural activity',
#                                 markerfacecolor='w', markersize=3.6,
#                                 markeredgewidth=.8),
#                   mlines.Line2D([0], [0], marker='s', color='k', label='action duration',
#                                 markerfacecolor='w', markersize=2.8,
#                                 markeredgewidth=.8)
#                  ]
#ax.legend(handles=legend_elements, title='decoder', loc='center')
#ax.axis('off')


#%% svm prediction plots
prob_value_df = (P.set_index(['shuffled','genotype','animal','date','label','actionNo'])
                  .loc[False, ['action','o!','r.','noNeurons']])
prob_value_df['value'] = (actionValues.set_index(['genotype','animal','date','label','actionNo'])
                                      .value)
prob_value_df = prob_value_df.reset_index()
prob_value_df['stay'] = prob_value_df.label.str.endswith('.').astype('int')

    
##%%
data = prob_value_df.query('action in ["mL2C","mR2C"]').dropna().copy()
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
    #ax.invert_xaxis()
    if gt == 'a2a':
        ax.set_xlabel('action value')
    ax.set_yticks((0,.5,1))
    if gt == 'd1':
        #ax.set_yticklabels((0,50,100))
        ax.set_yticklabels((-100, 0, 100))
        #ax.set_ylabel('SVM prediction\nP(win-stay)')
        #ax.set_ylabel('certainty')
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
    axkde.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[gt])


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
            ax.set_ylabel('r(action value*, certainty)')
            ax.set_yticklabels((.0,.5))
        sns.despine(ax=ax, bottom=True, trim=True)
    else:
        ax.set_axis_off()
 
    
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecoding.svg")


#%%##############################################################################
##%% phase stay-switch auc correlations
##def shuffleTuning(actionTuning):
##    return pd.Series(np.random.permutation(actionTuning),
##                     index=actionTuning.index)
##    
##def getPercentile(value, shuffle_dist):
##    return np.searchsorted(np.sort(shuffle_dist), value) / len(shuffle_dist)
#
#
#order = ['mL2C','dL2C','mC2L','pC2L','pC2R','mC2R','dR2C','mR2C']
#tunings = staySwitchAUC.set_index(['genotype','animal','date','neuron','action']).auc
#
#similarity = tunings.abs().unstack()[order].groupby('genotype').corr().stack()
#
##similarity_shuffled = []
##for _ in range(1000):
##    tunings_shuffled = (tunings.groupby(['genotype','animal','date','action'])
##                               .apply(shuffleTuning))
##    similarity_shuffled.append((tunings_shuffled.unstack()[order]
##                                                .groupby('genotype')
##                                                .corr().stack()))
##similarity_shuffled = pd.concat(similarity_shuffled, axis=1,
##                                keys=np.arange(len(similarity_shuffled)),
##                                names=['shuffle_no'])
##
##percentile = similarity_shuffled.apply(lambda g: getPercentile(similarity.loc[g.name], g),
##                                       axis=1)   
#
###%%
#vmin, vmax = -1, 1
#cmap = cmocean.cm.balance #cmocean.cm.thermal #cmocean.cm.amp
#
#for genotype in ("oprm1", "d1", "a2a"):
#    ax = layout.axes['{}_corr_m'.format(genotype)]['axis']
#    
#    corr = similarity.loc[genotype].unstack(-1).values
##    pctl = percentile.loc[genotype].unstack(-1).values
#    
##    sign_matrix = np.array(([''] * 36)).reshape(6,6)
##    sign_matrix[(pctl < .005) | (pctl > .995)] = '*'
#    
#    corr[np.triu_indices_from(corr)] = np.nan
#    corr = np.ma.masked_where(np.isnan(corr), corr)
##    sign_matrix = np.ma.masked_where(np.isnan(corr), sign_matrix)
#
#    cm = ax.pcolormesh(corr, cmap=cmap, vmin=vmin, vmax=vmax,
#                       edgecolors='w', lw=.35)
##    for y,x in zip(*np.tril_indices_from(sign_matrix, -1)):
##        ax.text(x+.5,y+.22, sign_matrix[(y, x)], ha='center', va='center',
##                color='k', fontsize=7)
#    ax.set_xlim((0,6))
#    ax.set_ylim((1,7))
#    ax.axis('off')
#      
#    pal = [style.getColor(a) for a in order]
#    n = len(pal)
#    for orient in ('v','h'):
#        ax = layout.axes['{}_{}cbar'.format(genotype, orient)]['axis']
#        ax.pcolormesh(np.arange(n).reshape(1,n) if orient=='h' 
#                          else np.arange(n).reshape(n,1),
#                      cmap=mpl.colors.ListedColormap(list(pal)),
#                      alpha=1, edgecolors='w', lw=.35)
#        if orient=='h':
#            ax.set_xlim((0,6))
#        else:
#            ax.set_ylim((1,7))
#        ax.axis('off')
#
#cax = layout.axes['corr_colorbar']['axis']
#cb = plt.colorbar(cm, cax=cax, orientation='horizontal',
#                  ticks=(vmin, (vmin+vmax)/2, vmax))
#cax.tick_params(axis='x', which='both',length=0)
#cb.outline.set_visible(False)
#

##%%
#action_aucs = staySwitchAUC.query('action == @action').copy()
#action_aucs['stay'] = action_aucs.pct > .995
#action_aucs['switch'] = action_aucs.pct < .005
#
#sign_sess_frac = pd.DataFrame(action_aucs.groupby(['genotype','animal','date'])[['stay']].sum())
#sign_sess_frac['switch'] = action_aucs.groupby(['genotype','animal','date']).switch.sum()
#sign_sess_frac['noNeurons'] = action_aucs.groupby(['genotype','animal','date']).size()
#sign_sess_frac.loc[:,['stay','switch']] = (sign_sess_frac[['stay','switch']] / 
#                                           sign_sess_frac.noNeurons.values[:,np.newaxis])
#sign_sess_frac.reset_index(inplace=True)
#
#for tuning in ['stay','switch']:
#    fig = plt.figure(figsize=(1,1))
#    ax = plt.gca()
#    sessionBarPlot.sessionBarPlot(sign_sess_frac, tuning, ax, style.getColor,
#                                  weightScale=.02)
#    ax.set_ylim((0,.4))
#    ax.set_yticks((0,.2,.4))
#    ax.set_yticks((.1, .3), minor=True)
#    ax.set_yticklabels((ax.get_yticks()*100).astype('int'))
#    sns.despine(ax=ax)
#    plt.show()
    
    
#%%
#stay_corrs = analysisStaySwitchDecoding.getActivityCorrs(endoDataPath, stay_aucs,
#                                                         actionValues, action)
#switch_corrs = analysisStaySwitchDecoding.getActivityCorrs(endoDataPath, switch_aucs,
#                                                           actionValues, action)
#corrs = pd.concat([stay_corrs,switch_corrs], axis=0, keys=['stay','switch'], names=['tuning'])
#
##%%
#corrs['diff'] = corrs.value.abs() - corrs.duration.abs()
#corrs.reset_index(inplace=True)
#
##%%
#fig = plt.figure(figsize=(1,1.5))
#ax = plt.gca()
#vs = sns.violinplot(x='genotype', y='diff', data=corrs, order=['d1','a2a','oprm1'],
#                    palette=[style.getColor(g) for g in ['d1','a2a','oprm1']], inner='box',
#                    linewidth=.5, ax=ax, scale='area')
#plt.setp(vs.axes.collections, alpha=.5, linewidth=0)
#ax.axhline(0, ls=':', c='k', alpha=1, lw=mpl.rcParams['axes.linewidth'])
#ax.set_ylabel('|R(value)| - |R(duration)|')
#ax.set_xlabel('')
#ax.set_xticks(())
#sns.despine(ax=ax, trim=True, bottom=True)
#fig.savefig('svg/stsw_tuned_value-velo_corrs.svg', bbox_inches='tight', pad_inches=.1)
#plt.show()


#%%
#stay_corrs_means = stay_corrs.groupby(['genotype','animal','date']).mean()
#stay_corrs_means['noNeurons'] = stay_corrs.groupby(['genotype','animal','date']).size()
#stay_corrs_means.reset_index(inplace=True)
#switch_corrs_means = switch_corrs.groupby(['genotype','animal','date']).mean()
#switch_corrs_means['noNeurons'] = switch_corrs.groupby(['genotype','animal','date']).size()
#switch_corrs_means.reset_index(inplace=True)
##%%
#fig, axs = plt.subplots(1, 2, figsize=(1.5,1.5), sharey=True, sharex=True)
#for ax, corrs in zip(axs, [stay_corrs_means, switch_corrs_means]):
#    for gt, gcorrs in corrs.groupby('genotype'):
#        x = np.ones((2,len(gcorrs)))
#        x[0,:] = 0
#        x += np.random.normal(0,.14,size=x.shape[1])
#        
#        ax.scatter(x, [gcorrs.duration, gcorrs.value], s=gcorrs.noNeurons / 10.,
#                   c=[style.getColor(gt),], linewidths=0, alpha=.4, clip_on=False)
#    ax.axhline(0, ls=':', c='k', zorder=-99, alpha=.35)
#    ax.set_xticks((0,1))
#    ax.set_xticklabels(['action duration','action value'], rotation=30,
#                       ha='right')
#    ax.set_xlim((-.5,1.5))
#    ax.set_ylim((-.4,.4))
#    ax.set_yticks((-.4,-.2,0,.2,.4))
#    ax.set_yticks((-.3,-.1,.1,.3), minor=True)
#    sns.despine(ax=ax)
#axs[0].set_ylabel('correlation')
#fig.savefig('svg/duration_value_corrs.svg', bbox_inches='tight', pad_inches=.1)
#plt.show()

