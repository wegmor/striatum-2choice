import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import figurefirst
import cmocean
import tqdm
from matplotlib.ticker import MultipleLocator
from utils import fancyViz
from utils import readSessions
import analysisTunings
import style
import subprocess

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
svgName = "figureS4PhaseTuning.svg"
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()


#%% Panel A
selection = pd.read_csv("continuitySelection.csv", comment="#")
fancyVizs = {}
signals = {}
lw = mpl.rcParams["axes.linewidth"]
uniqueSessions = selection.session.unique()
with tqdm.tqdm(total=len(uniqueSessions), desc="Loading data") as t:
    for sess in readSessions.findSessions(endoDataPath, task="2choice"):
        if str(sess) in uniqueSessions:
            k = str(sess)
            neurons = selection[selection.session == str(sess)].neuron
            fancyVizs[k] = fancyViz.SchematicIntensityPlot(sess, smoothing=7, linewidth=lw, splitReturns=False,
                                                           saturation=1)
            signals[k] = sess.readDeconvolvedTraces(rScore=True)[neurons]
            t.update(1)

for i, (session, neuron) in selection.iterrows():
    if i>=14: break
    ax = layout.axes["s{}".format(i+1)]["axis"]
    img = fancyVizs[session].draw(signals[session][neuron], ax=ax)
    genotype, animal, date = session.split("_")
    ax.set_title("#{} ({})\nneuron {}".format(animal, date, neuron), fontsize=7)

cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, -.7, 'z-score', ha='center', va='top', fontdict={'fontsize':6})

    
#%% Panel B
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
            ax.set_title({'d1':'D1+','a2a':'A2A+','oprm1':'Oprm1+'}[g])
        ax.set_ylabel('')
        ax.set_yticks((0,.5,1))
        ax.set_yticklabels(())
        if g == 'd1':
            ax.set_ylabel('{} tuned (%)'.format({'signp':'positively','signn':'negatively'}[tuning]))
            ax.set_yticklabels((0,50,100))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_ylim((0,1))
        
        sns.despine(ax=ax)


#%% Panel C
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
    ax.set_title({'d1':'D1+','a2a':'A2A+','oprm1':'Oprm1+'}[genotype])


#%% Panel E
# df = analysisTunings.getPDistVsCorrData(endoDataPath)

# df['bin'] = pd.cut(df.dist_orig, np.logspace(np.log10(15), 3, 10))
# df['bin_perm'] = pd.cut(df.dist_perm, np.logspace(np.log10(15), 3, 10))

# df_orig = (df.groupby(['genotype','animal','date','bin'])[['dist_orig','cc']]
#              .mean().dropna())
# df_perm = (df.groupby(['genotype','animal','date','bin_perm'])[['dist_perm','cc']]
#              .mean().dropna())
# no_neurons = df.groupby(['genotype','animal','date']).noNeurons.first()

# for g, data in df_orig.groupby('genotype'):
#     data_perm = df_perm.loc[g].copy()
#     ax = layout.axes['{}_dist_corr'.format(g)]['axis']
    
#     for (a,d), adata in data.groupby(['animal','date']):
#         ax.plot(adata.unstack('bin').dist_orig.T, adata.unstack('bin').cc.T,
#                 color=style.getColor(g), alpha=.2, #lw=mpl.rcParams['axes.linewidth'])
#                 lw=no_neurons.loc[(g,a,d)]/200)
    
#     data = data.copy()
#     data['noNeurons'] = no_neurons
#     data_perm['noNeurons'] = no_neurons.loc[g]
    
#     avg = pd.concat([data.groupby('bin').apply(analysisTunings.wAvg,
#                                                'cc', 'noNeurons'),
#                      data.groupby('bin').apply(analysisTunings.wAvg,
#                                                'dist_orig', 'noNeurons')],
#                     axis=1, keys=['cc','dist'])
#     sem = pd.concat([data.groupby('bin').apply(analysisTunings.bootstrap,
#                                                'cc', 'noNeurons'),
#                      data.groupby('bin').apply(analysisTunings.bootstrap,
#                                                'dist_orig', 'noNeurons')],
#                     axis=1, keys=['cc','dist'])  
#     ax.errorbar(x=avg.dist, y=avg.cc, xerr=sem.dist, yerr=sem.cc,
#                 color=style.getColor(g))
    
#     avg_perm = pd.concat([data_perm.groupby('bin_perm').apply(analysisTunings.wAvg,
#                                                               'cc', 'noNeurons'),
#                           data_perm.groupby('bin_perm').apply(analysisTunings.wAvg,
#                                                               'dist_perm', 'noNeurons')],
#                           axis=1, keys=['cc','dist'])
#     sem_perm = pd.concat([data_perm.groupby('bin_perm').apply(analysisTunings.bootstrap,
#                                                               'cc', 'noNeurons'),
#                           data_perm.groupby('bin_perm').apply(analysisTunings.bootstrap,
#                                                               'dist_perm', 'noNeurons')],
#                           axis=1, keys=['cc','dist'])
#     ax.errorbar(x=avg_perm.dist, y=avg_perm.cc, xerr=sem_perm.dist, yerr=sem_perm.cc,
#                 color=style.getColor('shuffled'), zorder=-99)
    
#     ax.set_xscale('log')
#     ax.set_xlim((10,1000))
#     ax.set_xticks((10,100,1000))
#     ax.set_ylim((-.025, .3))
#     ax.set_ylabel('')
#     ax.set_yticks((0,.1,.2,.3))
#     ax.yaxis.set_minor_locator(MultipleLocator(.05))
#     ax.set_yticklabels(())
#     if g == 'd1':
#         ax.set_ylabel('correlation')
#         ax.set_yticklabels(ax.get_yticks())
#     ax.set_xlabel('')
#     if g == 'a2a':
#         ax.set_xlabel('distance (μm)')

#     ax.set_title({'d1':'D1+','a2a':'A2A+','oprm1':'Oprm1+'}[g])
#     sns.despine(ax=ax)
   
    
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)
subprocess.check_call(['inkscape', outputFolder / svgName,
                           '--export-pdf={}pdf'.format(outputFolder / svgName[:-3])])
