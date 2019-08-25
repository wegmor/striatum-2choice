import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pathlib
import figurefirst
import cmocean
import tqdm
from matplotlib.ticker import MultipleLocator

#import sys
#thisFolder = pathlib.Path(__file__).resolve().parent
#sys.path.append(str(thisFolder.parent))

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisTunings
import style

style.set_context()
plt.ioff()


#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder = pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()


#%%
layout = figurefirst.FigureLayout(templateFolder / "tuningsSupp.svg")
layout.make_mplfigures()


#%% Panel A
cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisTunings.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)

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


order = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co", "pL2Cr",
         "pC2L-", "pC2R-", "pR2Cd", "pR2Co", "pR2Cr"]

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


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "tuningsSupp.svg")

