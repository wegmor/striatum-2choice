import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import figurefirst
import cmocean
import tqdm
#import os
from matplotlib.ticker import MultipleLocator
from collections import defaultdict
from utils import fancyViz
from utils import readSessions
import analysisTunings
import style

style.set_context()
plt.ioff()


#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "tuningsSupp1.svg")
layout.make_mplfigures()

#%% Panel A
tuningData = analysisTunings.getTuningData(endoDataPath)

df = tuningData.copy()

df['signp'] = df['pct'] > .995
df['signn'] = df['pct'] < .005
df['sign'] = df.signp.astype('int') - df.signn.astype('int')

sign_count = (df.groupby(['genotype','animal','date','action'])
                .agg({'signp':'sum','signn':'sum'}))
total_count = (df.groupby(['genotype','animal','date','action'])
                 [['signp','signn']].count())
sign_pct = sign_count / total_count
sign_pct['noNeurons'] = total_count.signp


order = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "dL2C-", "pL2Co", "pL2Cr",
         "pC2L-", "pC2R-", "dR2C-", "pR2Co", "pR2Cr"]

# v x coords for actions
a2x = dict(zip(order, np.arange(.5,12)))
sign_pct['x'] = sign_pct.reset_index().action.replace(a2x).values
# v color for actions
sign_pct['color'] = sign_pct.reset_index().action.str.slice(0,4).apply(style.getColor).values

for tuning in ('signp','signn'):
    for g, gdata in sign_pct.groupby('genotype'):
        ax = layout.axes['{}_{}_tuning'.format(g,{'signp':'pos','signn':'neg'}[tuning])]['axis']
        ax.scatter(analysisTunings.jitter(gdata.x, .15), gdata[tuning], s=gdata.noNeurons/20,
                   edgecolor=gdata.color, facecolor='none', clip_on=False)
        
        avg = gdata.groupby('x').apply(analysisTunings.wAvg, tuning, 'noNeurons')
        sem = gdata.groupby('x').apply(analysisTunings.bootstrap, tuning, 'noNeurons')
        ax.errorbar(avg.index, avg, sem, fmt='.-', c='k')
        
    #    for a, adata in gdata.groupby('animal'):
    #        avg = adata.groupby('x').apply(analysisTunings.wAvg, tuning, 'noNeurons')
    #        lw = adata.groupby('date').noNeurons.first().sum() / 500
    #        ax.plot(avg, c='k', alpha=.35, lw=lw)
        
        ax.set_xticks(np.arange(.5,12))
        ax.set_xlim((0,12))
        ax.set_xticklabels(())
        if tuning == 'signp':
            ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])
        ax.set_ylabel('')
        ax.set_yticks((0,.5,1))
        ax.set_yticklabels(())
        if g == 'd1':
            ax.set_ylabel('{} tuned (%)'.format({'signp':'positively','signn':'negatively'}[tuning]))
            ax.set_yticklabels((0,50,100))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_ylim((0,1))
        
        sns.despine(ax=ax)


#%% Panel B
tunings = tuningData.set_index(["genotype", "animal", "date", "neuron", "action"]).tuning

cax = layout.axes['corr_colorbar']['axis']
cax.tick_params(axis='y', which='both',length=0)

for genotype in ("oprm1", "d1", "a2a"):
    corr = tunings.loc[genotype].unstack()[order].corr()
    ax = layout.axes["correlationMatrix_{}".format(genotype)]["axis"]
    hm = sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, annot=True, fmt=".2f",
                     cmap=cmocean.cm.balance, cbar=True, cbar_ax=cax,
                     cbar_kws={'ticks':(-1,0,1)}, xticklabels=False, yticklabels=False,
                     annot_kws={'fontsize': 4.0},
                     linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[genotype])


#%% Panel C
saturation = .2
meanPlots = {g: fancyViz.SchematicIntensityPlot(splitReturns=False,
                                                linewidth=mpl.rcParams['axes.linewidth'],
                                                smoothing=5.5,
                                                saturation=saturation)
                 for g in ("oprm1", "d1", "a2a")}

for sess in readSessions.findSessions(endoDataPath, task="2choice"):
    signal = sess.readDeconvolvedTraces(zScore=True)
    if len(signal) != len(sess.readSensorValues()):
        continue
    genotype = sess.meta.genotype
    meanPlots[genotype].setSession(sess)
    for neuron in tqdm.tqdm(signal.columns, desc=str(sess)):
        meanPlots[genotype].addTraceToBuffer(signal[neuron])
        
for genotype, meanPlot in meanPlots.items():
    ax = layout.axes["genotypeAvg_{}".format(genotype)]["axis"]
    img = meanPlot.drawBuffer(ax=ax)
    
cax = layout.axes['avgs_colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
#cax.xaxis.tick_top()
#cax.tick_params(axis='both', which='both',length=0)
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-.025, .25, -saturation, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, saturation, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)


#%% Panel D
df = analysisTunings.getTaskNoTaskData(endoDataPath)
df['unengaged'] -= df.engaged

for g, gdata in df.groupby('genotype'):
    ax = layout.axes['{}_task_nontask'.format(g)]['axis']
    
    for u,n in gdata[['unengaged','noNeurons']].values:
        ax.plot([0,u], color=style.getColor(g),
                alpha=.2, lw=n/200, clip_on=False)
    
    avg = analysisTunings.wAvg(gdata, 'unengaged', 'noNeurons')
    sem = analysisTunings.bootstrap(gdata, 'unengaged', 'noNeurons')
    ax.errorbar(0, 0, yerr=0, color=style.getColor(g), marker='o',
                markersize=3.2, markeredgewidth=mpl.rcParams['lines.linewidth'],
                clip_on=False)
    ax.errorbar(1, avg, yerr=sem, color=style.getColor(g), marker='v',
                markersize=3.6, markeredgewidth=mpl.rcParams['lines.linewidth'],
                clip_on=False)
    ax.plot([0,1], [0,avg], color=style.getColor(g))
    
    ax.axhline(0, ls=':', lw=mpl.rcParams['axes.linewidth'],
               c='k', alpha=.5, zorder=-99)
    
    ax.set_xticks(())
    #ax.set_xticklabels(['task','non-task'], rotation=40, ha='right')
    ax.set_ylim((0,.2))
    ax.set_yticks((0,.1,.2))
    ax.set_yticklabels(())
    ax.set_xlim((-.2,1.2))
    ax.axis('off')
    if g == 'd1':
        ax.axis('on')
        ax.set_ylabel('Δsd')
        ax.set_yticklabels(ax.get_yticks())
        sns.despine(ax=ax, bottom=True)
    ax.set_title({'d1':'D1', 'a2a':'A2A', 'oprm1':'Oprm1'}[g])
        
#axt = layout.axes['task_notask_legend1']['axis']
#legend_elements = [mlines.Line2D([0], [0], marker='o', color='k', label='task',
#                                 markerfacecolor='k', markersize=3.2,
#                                 markeredgewidth=mpl.rcParams['lines.linewidth']),
#                   mlines.Line2D([0], [0], marker='v', color='k', label='non-task',
#                                 markerfacecolor='k', markersize=3.6,
#                                 markeredgewidth=mpl.rcParams['lines.linewidth']),
#                  ]
#axt.legend(handles=legend_elements, loc='center', ncol=2, mode='expand')
#axt.axis('off')

#axt = layout.axes['task_notask_legend2']['axis']
#legend_elements = [mlines.Line2D([0], [0], color=style.getColor(gt),
#                                 label={'d1':'D1', 'a2a':'A2A', 'oprm1':'Oprm1'}[gt])
#                       for gt in ['d1','a2a','oprm1']
#                  ]
#axt.legend(handles=legend_elements, loc='center', ncol=3, mode='expand')
#axt.axis('off')


#%% Panel E
df = analysisTunings.getPDistVsCorrData(endoDataPath)

df['bin'] = pd.cut(df.dist_orig, np.logspace(np.log10(15), 3, 10))
df['bin_perm'] = pd.cut(df.dist_perm, np.logspace(np.log10(15), 3, 10))

df_orig = (df.groupby(['genotype','animal','date','bin'])[['dist_orig','cc']]
             .mean().dropna())
df_perm = (df.groupby(['genotype','animal','date','bin_perm'])[['dist_perm','cc']]
             .mean().dropna())
no_neurons = df.groupby(['genotype','animal','date']).noNeurons.first()

for g, data in df_orig.groupby('genotype'):
    data_perm = df_perm.loc[g].copy()
    ax = layout.axes['{}_dist_corr'.format(g)]['axis']
    
    for (a,d), adata in data.groupby(['animal','date']):
        ax.plot(adata.unstack('bin').dist_orig.T, adata.unstack('bin').cc.T,
                color=style.getColor(g), alpha=.2, #lw=mpl.rcParams['axes.linewidth'])
                lw=no_neurons.loc[(g,a,d)]/200)
    
    data = data.copy()
    data['noNeurons'] = no_neurons
    data_perm['noNeurons'] = no_neurons.loc[g]
    
    avg = pd.concat([data.groupby('bin').apply(analysisTunings.wAvg,
                                               'cc', 'noNeurons'),
                     data.groupby('bin').apply(analysisTunings.wAvg,
                                               'dist_orig', 'noNeurons')],
                    axis=1, keys=['cc','dist'])
    sem = pd.concat([data.groupby('bin').apply(analysisTunings.bootstrap,
                                               'cc', 'noNeurons'),
                     data.groupby('bin').apply(analysisTunings.bootstrap,
                                               'dist_orig', 'noNeurons')],
                    axis=1, keys=['cc','dist'])  
    ax.errorbar(x=avg.dist, y=avg.cc, xerr=sem.dist, yerr=sem.cc,
                color=style.getColor(g))
    
    avg_perm = pd.concat([data_perm.groupby('bin_perm').apply(analysisTunings.wAvg,
                                                              'cc', 'noNeurons'),
                          data_perm.groupby('bin_perm').apply(analysisTunings.wAvg,
                                                              'dist_perm', 'noNeurons')],
                          axis=1, keys=['cc','dist'])
    sem_perm = pd.concat([data_perm.groupby('bin_perm').apply(analysisTunings.bootstrap,
                                                              'cc', 'noNeurons'),
                          data_perm.groupby('bin_perm').apply(analysisTunings.bootstrap,
                                                              'dist_perm', 'noNeurons')],
                          axis=1, keys=['cc','dist'])
    ax.errorbar(x=avg_perm.dist, y=avg_perm.cc, xerr=sem_perm.dist, yerr=sem_perm.cc,
                color=style.getColor('shuffled'), zorder=-99)
    
    ax.set_xscale('log')
    ax.set_xlim((10,1000))
    ax.set_xticks((10,100,1000))
    ax.set_ylim((-.025, .3))
    ax.set_ylabel('')
    ax.set_yticks((0,.1,.2,.3))
    ax.yaxis.set_minor_locator(MultipleLocator(.05))
    ax.set_yticklabels(())
    if g == 'd1':
        ax.set_ylabel('correlation')
        ax.set_yticklabels(ax.get_yticks())
    ax.set_xlabel('')
    if g == 'a2a':
        ax.set_xlabel('distance (μm)')

    ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])
    sns.despine(ax=ax)

#%% Pie charts with only largest (most neurons) session from each animal
df = tuningData.copy()

df['signp'] = df['pct'] > .995
df['signn'] = df['pct'] < .005
df['sign'] = df.signp.astype('int') - df.signn.astype('int')

nNeurons = df.groupby(["animal", "date"]).neuron.nunique()
largestSessions = nNeurons.groupby(level=0).idxmax()
df_ls = df.set_index(["animal", "date"]).loc[largestSessions].reset_index()

# only keep max tuning for each neuron
maxdf = df_ls.loc[df_ls.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~maxdf.signp, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

# create dictionary with modified alpha to separate r/o/d phases
cdict = defaultdict(lambda: np.array([1,1,1]),
                    {a:style.getColor(a[:4]) for a 
                     in ['mC2L-','mC2R-','mL2C-','mR2C-','pC2L-','pC2R-','pL2C-','pR2C-']})
cdict['pL2Cr'] = cdict['pL2C-']
cdict['pL2Co'] = np.append(cdict['pL2C-'], .45)
cdict['dL2C-'] = np.append(cdict['pL2C-'], .7)
cdict['pR2Cr'] = cdict['pR2C-']
cdict['pR2Co'] = np.append(cdict['pR2C-'], .45)
cdict['dR2C-'] = np.append(cdict['pR2C-'], .7)
cdict['pC2L-'] = np.append(cdict['pC2L-'], .45)

for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']

    gdata = maxdf.loc[g].loc[['mC2R-','mL2C-','mC2L-','mR2C-','none',
                              'dL2C-','pL2Co','pL2Cr',
                              'dR2C-','pR2Co','pR2Cr',
                              'pC2L-','pC2R-',]]
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[cdict[a] for a in gdata.index])

    ax.set_aspect('equal')
   
    
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "tuningsSupp1.svg")


#%% Supp 2
layout = figurefirst.FigureLayout(templateFolder / "tuningsSupp2.svg")
layout.make_mplfigures()

#%% TSNE 
tuningTsne = analysisTunings.getTSNEProjection(tuningData)

#%% tuning data
tuningData = analysisTunings.getTuningData(endoDataPath)
    
#%% "intensity color code" tsne by tuning
tsne = tuningTsne.set_index(['genotype','animal','date','neuron'])[[0,1]].copy()
tunings = tuningData.set_index(['genotype','animal','date','neuron'])[['action','tuning']].copy()
df = (pd.merge(tunings, tsne, how='left', left_index=True, right_index=True)
        .dropna().set_index('action', append=True))

for action, adata in df.groupby('action'):
    ax = layout.axes['{}_tsne'.format(action)]['axis']
    sca = ax.scatter(adata[0], adata[1], c=adata['tuning'],
                     marker='.', alpha=.75, s=5, lw=0, clip_on=False,
                     cmap=cmocean.cm.balance, vmin=-6, vmax=6,
                     rasterized=True)
    
    ax.set_xlim((adata[0].min(), adata[0].max()))
    ax.set_ylim((adata[1].min(), adata[1].max()))
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    for gt, gdata in adata.groupby('genotype'):
        ax = layout.axes['{}_tsne_{}'.format(action, gt)]['axis']
        ax.scatter(gdata[0], gdata[1], c=gdata['tuning'],
                   marker='.', alpha=.75, s=1.5, lw=0, clip_on=False,
                   cmap=cmocean.cm.balance, vmin=-6, vmax=6,
                   rasterized=True)
        
        ax.set_xlim((adata[0].min(), adata[0].max()))
        ax.set_ylim((adata[1].min(), adata[1].max()))
        ax.invert_xaxis()
        ax.set_aspect('equal')
        if action in ['pR2Cd', 'pR2Co', 'pR2Cr']:
            ax.set_xlabel({'d1':'D1', 'a2a':'A2A', 'oprm1':'Oprm1'}[gt], labelpad=3.4)
        else:
            ax.set_xlabel('')
        ax.set_xticks(())
        ax.set_yticks(())
        sns.despine(top=True, bottom=True, left=True, right=True, ax=ax)
    
    ax = layout.axes['{}_dist'.format(action)]['axis']
    for gt, gdata in adata.groupby('genotype'):
        sns.kdeplot(gdata.tuning, ax=ax, color=style.getColor(gt),
                    clip_on=True, alpha=.75, label='', cut=2)
    ax.axvline(0, ls=':', color='darkgray', lw=mpl.rcParams['axes.linewidth'], alpha=1,
               zorder=-99)

    ax.set_xlim((-18,18))
    ax.set_ylim((-.005,.23))
    ax.set_yticks((0,.1,.2))
    ax.set_yticks((.05,.15), minor=True)
    ax.set_xticks(())
    ax.set_xlabel('')
    if action in ['mL2C-', 'pL2Cd', 'mR2C-', 'pR2Cd']:
        ax.set_ylabel('density')
    else:
        ax.set_ylabel('')
    if action == 'mL2C-':
        ax.legend(['A2A','D1','Oprm1'], loc='upper left', bbox_to_anchor=(.55,.95))
    sns.despine(bottom=True, trim=True, ax=ax)
    
    ax = layout.axes['{}_bp'.format(action)]['axis']
    palette = {gt: style.getColor(gt) for gt in ['d1','a2a','oprm1']}
    sns.boxplot('tuning', 'genotype', data=adata.reset_index('genotype'), ax=ax, 
                palette=palette, saturation=.85, showcaps=False, showfliers=False,
                boxprops={'alpha':0.75, 'lw':0, 'zorder':-99, 'clip_on':False}, 
                width=.75, whiskerprops={'c':'k','zorder':99, 'clip_on':False},
                medianprops={'c':'k','zorder':99, 'clip_on':False},
                order=['d1','a2a','oprm1'])
    ax.axvline(0, ls=':', color='darkgray', lw=mpl.rcParams['axes.linewidth'], alpha=1,
               zorder=-99)
    
    ax.set_xlim((-18,18))
    ax.set_ylim((-.75,2.75))
    ax.set_xticks((-12,0,12))
    ax.set_xticks((-6,6), minor=True)
    if action in ['pR2Cd', 'pR2Co', 'pR2Cr']:
        ax.set_xlabel('tuning score')
    else:
        ax.set_xlabel('')
    ax.set_yticks(())
    ax.set_ylabel('')
    sns.despine(left=True, trim=True, ax=ax)

cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(sca, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-0.05, 0.5, -6, ha='right', va='center',
         fontdict={'fontsize':6}, transform=cax.transAxes)
cax.text(1.05, 0.5, 6, ha='left', va='center',
         fontdict={'fontsize':6}, transform=cax.transAxes)
cax.text(0.5, -.1, 'tuning score', ha='center', va='top',
         fontdict={'fontsize':6}, transform=cax.transAxes)

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "tuningsSupp2.svg")
#os.system('convert -density 300 {} {}'.format(outputFolder / 'tuningsSupp2.svg',
#                                              outputFolder / 'tuningsSupp2.jpg'))
