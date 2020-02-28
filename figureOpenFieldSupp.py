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
from matplotlib.ticker import MultipleLocator

import analysisOpenField, analysisTunings
import style
from utils import readSessions, fancyViz

style.set_context()
plt.ioff()

#%%

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "openFieldSupp.svg")
layout.make_mplfigures()

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}

#All 4-behavior panels    
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)

## Panel A
tuningData = analysisOpenField.getTuningData(endoDataPath, segmentedBehavior)
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

order = ["stationary", "running", "leftTurn", "rightTurn"]

# v x coords for actions
a2x = dict(zip(order, np.arange(.5,12)))
sign_pct['x'] = sign_pct.reset_index().action.replace(a2x).values
# v color for actions
sign_pct['color'] = sign_pct.reset_index().action.apply(style.getColor).values

for tuning in ('signp','signn'):
    for g, gdata in sign_pct.groupby('genotype'):
        ax = layout.axes['fracTuned_{}_{}'.format(g,{'signp':'pos','signn':'neg'}[tuning])]['axis']
        ax.scatter(analysisOpenField.jitter(gdata.x, .15), gdata[tuning], s=gdata.noNeurons/20,
                   edgecolor=gdata.color, facecolor='none', clip_on=False)
        
        avg = gdata.groupby('x').apply(analysisOpenField.wAvg, tuning, 'noNeurons')
        sem = gdata.groupby('x').apply(analysisOpenField.bootstrap, tuning, 'noNeurons')
        ax.errorbar(avg.index, avg, sem, fmt='.-', c='k')
        
        ax.set_xticks(np.arange(.5,4))
        ax.set_xlim((0,4))
        ax.set_xticklabels([behaviorNames[b] for b in order], rotation=45, ha="right")
        title = genotypeNames[g]
        title += "\n" + {'signp':'positively','signn':'negatively'}[tuning]
        ax.set_title(title)
        ax.set_ylabel('')
        ax.set_yticks((0,.5,1))
        ax.set_yticklabels(())
        if g == 'oprm1' and tuning == 'signp':
            ax.set_ylabel('tuned neurons (%)')
            ax.set_yticklabels((0,50,100))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_ylim((0,1))
        
        sns.despine(ax=ax)
        
## Panel B
tunings = tuningData.set_index(["genotype", "animal", "date", "neuron", "action"]).tuning

cax = layout.axes['corr_colorbar']['axis']
cax.tick_params(axis='x', which='both',length=0)

for genotype in ("oprm1", "d1", "a2a"):
    corr = tunings.loc[genotype].unstack()[order].corr()
    ax = layout.axes["corrMatrix_{}".format(genotype)]["axis"]
    yticks = [behaviorNames[b] for b in order] if genotype == "oprm1" else False
    hm = sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, annot=True, fmt=".2f",
                     cmap=cmocean.cm.balance, cbar=True, cbar_ax=cax,
                     cbar_kws={'ticks':(-1,0,1), 'orientation': 'horizontal'},
                     annot_kws={'fontsize': 4.0}, yticklabels=yticks,
                     xticklabels=[behaviorNames[b] for b in order],
                     linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(genotypeNames[genotype], pad=3)
    ax.set_ylim(4, 0)
    ax.tick_params("both", length=0, pad=3)

layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "openFieldSupp.svg")
