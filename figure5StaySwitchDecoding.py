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
from matplotlib.ticker import MultipleLocator, FixedLocator
import pathlib
import figurefirst
import style
import analysisStaySwitchDecoding
import analysisStaySwitchDecodingSupp
import analysisOftVs2Choice
from utils import readSessions, fancyViz, sessionBarPlot
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
svgName = 'figure5StaySwitchDecoding.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()


#%%
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
            ("5703", "190130", 28, "oprm1")]

for p, (a, d, n, gt) in enumerate(examples):
    s = next(readSessions.findSessions(endoDataPath, genotype=gt,
                                       animal=a, date=d, task='2choice'))
    traces = s.readDeconvolvedTraces(rScore=True)
    lfa = s.labelFrameActions(reward='fullTrial', switch=True).set_index(traces.index)
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
                                 label={'oprm1':'Oprm1+', 'a2a':'A2A+', 'd1':'D1+',
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
                                  weightScale=.0075)
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
#df = df.query('action not in ["pL2C","pR2C"]') # TODO: we might as well show these as well...
df['sign'] = (df.pct > .995) | (df.pct < .005)

# only keep max tuning for each neuron
maxdf = (df.loc[df.groupby(['genotype','animal','date','neuron'])
                  .auc.apply(lambda t: t.abs().idxmax())])
# inidcate whether stay or switch tuned
maxdf.loc[maxdf.tuning.apply(np.sign) == 1, 'action'] += 'r.'
maxdf.loc[maxdf.tuning.apply(np.sign) == -1, 'action'] += 'o!'
maxdf.loc[~maxdf.sign, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['sign']].count() # get counts

# create dictionary with modified alpha to separate center/side port phases
cdict = {a:style.getColor(a[:4]) for a in maxdf.reset_index().action.unique()}
cdict['none'] = np.array((1,1,1))
cdict['pC2Lr.'] = np.append(cdict['pC2Lr.'], .45)
cdict['pC2Lo!'] = np.append(cdict['pC2Lo!'], .45)
cdict['dL2Cr.'] = np.append(cdict['dL2Cr.'], .45)
cdict['dL2Co!'] = np.append(cdict['dL2Co!'], .45)
cdict['dR2Cr.'] = np.append(cdict['dR2Cr.'], .45)
cdict['dR2Co!'] = np.append(cdict['dR2Co!'], .45)

for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']

    order = ['mC2Ro!','mC2Rr.','mL2Co!','mL2Cr.',
             'mC2Lo!','mC2Lr.','mR2Co!','mR2Cr.',
             'none',
             'dL2Co!','dL2Cr.','pL2Co!','pL2Cr.','dR2Co!','dR2Cr.','pR2Co!','pR2Cr.',
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
    
    ax.set_title({'d1':'D1+','a2a':'A2A+','oprm1':'Oprm1+'}[g])
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
axs['a2a'].set_xlabel('number of phases')


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
                       color='k', alpha=.5, zorder=-99, linestyle=':', clip_on=False)
        aaxs[i].vlines([4.5,9.5], -.18, .5, linestyle=':', color='k', alpha=1, 
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
                                     ['D1+','A2A+','Oprm1+'])],
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
        axfv = layout.axes['oprm1_f8_ex{}_{}'.format(p+1, trialType)]['axis']
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
    
    # v: 15->"compression factor" transforming frames to value
    avgAx.hlines(-1, 2-(10/15), 2+(10/15), ls='-', color='k', lw=mpl.rcParams['axes.linewidth'],
                 clip_on=False)
    avgAx.text(2, -1.15, '1s', ha='center', va='top', color='k', fontsize=6)
    
    for ax in [regAx, avgAx]:
        ax.set_xlim((-1,5))
        ax.set_ylabel('z-score')
    regAx.set_title('right to center\nturn', y=.9, fontsize=7)
    regAx.set_ylim((-.75,6))
    avgAx.set_ylim((-.75,4.75))
    regAx.set_xticks(np.arange(-1,6))
    regAx.set_xticks(np.arange(-1,5,.5), minor=True)
    avgAx.set_xticks(())
    regAx.set_xlabel('action value')
    avgAx.set_xlabel('')
    avgAx.set_yticks((0,2,4))
    avgAx.set_yticks((1,3,), minor=True)
    regAx.set_yticks((0,2,4,6))
    regAx.set_yticks((1,3,5), minor=True)
    sns.despine(ax=regAx, trim=False)
    sns.despine(ax=avgAx, bottom=True, trim=True)


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
