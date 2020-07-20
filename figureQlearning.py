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
import analysisQlearning, analysisStaySwitchDecoding
import cmocean
import scipy.stats
from utils import readSessions, sessionBarPlot, fancyViz
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
layout = figurefirst.FigureLayout(templateFolder / "qLearning.svg")
layout.make_mplfigures()

#Copied from figure5. TODO: This should be written much briefer
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
    regAVs = pd.read_pickle(cachedDataPath)
else:
    regAVs = analysisStaySwitchDecoding.getActionValues(endoDataPath, logRegCoef)
    regAVs.to_pickle(cachedDataPath)
regAVs["session"] = regAVs.genotype + "_" + regAVs.animal + "_" + regAVs.date
regAVs = regAVs.set_index(["session", "actionNo"])[["label", "value"]]
    
qAVs = analysisQlearning.getQActionValues(endoDataPath, perAnimal=True)

ax = layout.axes['ex_trace']['axis']
exSess = next(readSessions.findSessions(endoDataPath, animal="5703", date="190130"))
lfa = exSess.labelFrameActions()
lfa = lfa.join(qAVs.loc[str(exSess)][["Q_actionValue"]], on="actionNo")
lfa = lfa.join(regAVs.loc[str(exSess)][["value"]], on="actionNo")
ax.plot(lfa.Q_actionValue, label="Q action value")
ax.plot(lfa.value, label="regression action value")
for port in ("pC", "pL", "pR"):
    ax.fill_between(lfa.index, -6, 6, lfa.label.str.startswith(port), alpha=.15,
                    color=style.getColor(port), lw=0)
ax.set_xlim(10000, 20000)
ax.set_ylim(-7, 7)
ax.set_xticks([])
rewards = lfa.loc[lfa.label.str.endswith('r').astype('int').diff()==1].index.values
for r in rewards:
    fancyViz.drawWaterDrop(ax, np.array([r, 6]), np.array([50, 1]),
                           facecolor='k')
    ax.axvline(r, .05, .85, lw=.5, ls='--', color='k')
ax.hlines(-7, 18800, 20000, clip_on=False, color='k')
ax.text(19400, -8, "1 min".format(1000/20), ha="center", va="top", fontsize=6)
sns.despine(ax=ax, bottom=True)
ax.legend(ncol=2, bbox_to_anchor=(0.75, 1.05, 0.25, 0.1), mode="expand")

q_choices = qAVs[qAVs.label.str.match("pC2[RL]")].copy()
q_choices["rightChoice"] = q_choices.label.str.match("pC2R")

bins = np.arange(-1,1.1,0.1)
binned = pd.cut(q_choices.Qr_minus_Ql, bins)
fracRight = q_choices.groupby(binned).rightChoice.mean()
xx = (bins[1:] + bins[:-1])/2
ax = layout.axes['diffQ']['axis']
ax.plot(xx, 100*fracRight, ".-")
ax.set_xlabel("$Q_R - Q_L$")
ax.set_ylabel("right choice (%)")
ax.set_ylim(0,100)
sns.despine(ax=ax)

bins = np.arange(-5.5,5.6)
binned = pd.cut(q_choices.Q_actionValue, bins)
fracRight = q_choices.groupby(binned).rightChoice.mean()
xx = (bins[1:] + bins[:-1])/2
ax = layout.axes['Qav']['axis']
ax.plot(xx, 100*fracRight, ".-")
ax.set_xlabel("Q action value")
ax.set_yticklabels([])
ax.set_ylim(0,100)
#ax.set_ylabel("right choice (%)")
sns.despine(ax=ax)

params = analysisQlearning.fitQParametersPerAnimal(endoDataPath)
yticks = {'alpha': [0, 0.25, 0.5], 'beta': [0, 5, 10],
          'bias': [-.5, 0, .5]}
for p in ("alpha", "beta", "bias"):
    ax = layout.axes['params_'+p]['axis']
    sns.swarmplot(y=params[p], ax=ax, size=2)
    ax.set_ylabel("")
    ax.set_yticks(yticks[p])
    ax.set_xticks([])
    ax.set_ylim(yticks[p][0], yticks[p][-1])
    ax.set_title(p)
    sns.despine(ax=ax, bottom=True)

ax = layout.axes['Qav_vs_regav']['axis']
mask = qAVs.label.str.match("pC2[RL]")
#Scatter plot
#ax.plot(qAVs.Q_actionValue[mask], regAVs.value[mask], 'k.',
#        markersize=.2, alpha=.5, rasterized=True)

#Line plot

sns.lineplot(qAVs.Q_actionValue[mask].round(0), regAVs.value[mask], sort=True,
             ci="sd", err_style="bars", ax=ax, marker='.', markersize=5,
             markeredgecolor="C0")

#Figurefirst axis has proportions 3:4
ax.set_xlim(-4/3*5, 4/3*5)
ax.set_ylim(-5, 5)
ax.set_xticks([-5, 0, 5])
ax.set_yticks([-5, 0, 5])
ax.set_xlabel("Q action value")
ax.set_ylabel("regression\naction value")
sns.despine(ax=ax)

ax = layout.axes['inset_corr']['axis']
animals = qAVs.index.get_level_values(0).str.split("_").str[1]
correlations = qAVs.assign(regAV=regAVs.value).groupby(animals)\
                   .apply(lambda x: x.Q_actionValue.corr(x.regAV))
sns.swarmplot(y=correlations, ax=ax, size=2)
ax.set_ylabel("correlation")
ax.set_ylim(0.5, 1.0)
ax.set_yticks((0.5, 0.75, 1.0))
ax.set_xticks([])
sns.despine(ax=ax, bottom=True)


P = pd.read_pickle(cacheFolder / "stsw_p.pkl") #TODO: call proper function
P = P.query("not shuffled")
P["session"] = P.genotype + "_" + P.animal + "_" + P.date
P['stay'] = P.label.str.endswith('.').astype('int')
P = P[['shuffled','genotype','animal','date','session','label','actionNo','action','o!','r.','noNeurons']]
P = P.join(qAVs.Q_actionValue, on=["session", "actionNo"])
data = P.query('action in ["mL2C","mR2C"]').dropna().copy()
data = data.loc[data.label.str.contains('o!$|o\.$|r\.$')]

for (gt,label), gdata in data.groupby(['genotype','action']):
    ax = layout.axes['{}_value_ost'.format(gt)]['axis']
    for tt in ['o!','r.','o.']:
        ttdata = gdata.loc[gdata.label.str.endswith(tt)].copy()
        ttdata['bin'] = pd.qcut(ttdata.Q_actionValue, 4).cat.codes
        ttdata = ttdata.groupby(['animal','date','bin'])[['noNeurons','Q_actionValue','r.']].mean()
        
        stsw_wAvg = (ttdata.groupby('bin')
                           .apply(analysisStaySwitchDecoding.wAvg,'r.','noNeurons'))
        stsw_wSem = (ttdata.groupby('bin')
                           .apply(analysisStaySwitchDecoding.bootstrap,'r.','noNeurons'))
        value_wAvg = (ttdata.groupby('bin')
                            .apply(analysisStaySwitchDecoding.wAvg,'Q_actionValue','noNeurons'))
        value_wSem = (ttdata.groupby('bin')
                            .apply(analysisStaySwitchDecoding.bootstrap,'Q_actionValue','noNeurons'))
        
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
        ax.set_xlabel('Q action value')
    ax.set_yticks((0,.5,1))
    if gt == 'd1':
        ax.set_yticklabels((0,50,100))
        #ax.set_yticklabels((-100, 0, 100))
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
    gdata['bin'] = pd.cut(gdata.Q_actionValue, bins=bins, labels=labels).astype('float')
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
    corr = ttdata.groupby(['genotype','animal','date'])[['r.','Q_actionValue']].corr()
    corr = pd.DataFrame(pd.Series(corr.unstack(-1)[('Q_actionValue','r.')],
                                  name='correlation'))
    corr['noNeurons'] = ttdata.groupby(['genotype','animal','date']).noNeurons.first()
    return corr

def randomShiftValue(ttdata):
    def shift(v):
        v = pd.Series(np.roll(v, np.random.randint(10,30) * np.random.choice([-1,1])),
                      index=v.index)
        return v
    
    ttdata = ttdata.copy()
    ttdata['Q_actionValue'] = ttdata.groupby(['genotype','animal','date'])['Q_actionValue'].apply(shift).copy()
    return ttdata
    

valueProbCorrs = pd.DataFrame()
for tt, ttdata in data.groupby(data.label.str.slice(-2)):
    ttdata = ttdata.copy()
    left_trials = ttdata.label.str.contains('L')
    ttdata.loc[left_trials, 'Q_actionValue'] = ttdata.loc[left_trials, 'Q_actionValue'] * -1
    
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
            ax.set_ylabel('r(Q action value*, P(win-stay))')
            ax.set_yticklabels((.0,.5))
        sns.despine(ax=ax, bottom=True, trim=True)
    else:
        ax.set_axis_off()
 

layout.insert_figures('plots')
layout.write_svg(outputFolder / "qLearning.svg")