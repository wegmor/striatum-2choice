import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import h5py
import pathlib
import figurefirst
import cmocean

#import sys
#thisFolder = pathlib.Path(__file__).resolve().parent
#sys.path.append(str(thisFolder.parent))

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisDecoding
import style

style.set_context()
plt.ioff

#%%

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "decodingSupp.svg")
layout.make_mplfigures()

#%% Panel A
decodingData = analysisDecoding.decodingConfusion(endoDataPath)

order = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co", "pL2Cr",
         "pC2L-", "pC2R-", "pR2Cd", "pR2Co", "pR2Cr"]
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]

cax = layout.axes['dec_colorbar']['axis']
cax.tick_params(axis='both', which='both',length=0)

for gt, data in decodingData.groupby("genotype"):
    weightedData = data.set_index(["true", "predicted"]).eval("occurences * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    ax = layout.axes["confusionMatrix_{}".format(gt)]["axis"]
    sns.heatmap(weightedData[order].reindex(order), ax=ax, vmin=0, vmax=1, annot=True, fmt=".0%",
                cmap=cmocean.cm.amp, xticklabels=False, yticklabels=False,
                annot_kws={'fontsize': 4.5},  cbar=True, cbar_ax=cax,
                cbar_kws={'orientation':'horizontal', 'ticks':()},
                linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
cax.text(-.025, .25, 0, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, 100, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.1, 'accuracy (%)', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)

#%% Panel B
decodingData = analysisDecoding.decodeWithSortedNeurons(endoDataPath)

plt.sca(layout.axes["decodingWithSortedNeurons"]["axis"])
for (strSess, ordering), df in decodingData.groupby(["session", "ordering"]):
    plt.plot(df.groupby("nNeurons").accuracy.mean(), color=style.getColor(ordering),
             alpha=0.15, lw=.35)
for ordering, df in decodingData.groupby("ordering"):
    oavg = df.groupby('nNeurons').accuracy.mean()
    osem = df.groupby('nNeurons').accuracy.sem()
    plt.plot(oavg, color=style.getColor(ordering), alpha=1.0)
    plt.fill_between(oavg.index, oavg-osem, oavg+osem, lw=0,
                     color=style.getColor(ordering), alpha=.2, zorder=-99)
    
order = ("descending", "ascending")
meanHandles = [mpl.lines.Line2D([], [], color=style.getColor(o), label=o) for o in order]
plt.legend(handles=meanHandles, bbox_to_anchor=(1,.28), loc='center right', ncol=1, title="sorting")

plt.ylim(0,1)
plt.xlim(0,200)
plt.xlabel("number of neurons")
plt.ylabel("decoding accuracy (%)")
plt.yticks((0,.5,1), (0,50,100))
plt.gca().set_yticks(np.arange(.25,1,.25), minor=True)
plt.xticks(np.arange(0,201,50))
plt.gca().set_xticks(np.arange(25,200,25), minor=True)
sns.despine(ax=plt.gca())
    
#%% Panel C
def calcCorr(df):
    r = scipy.stats.pearsonr(df.true, df.predicted)[0]
    return pd.Series((r, df.nNeurons.iloc[0]), ("correlation", "nNeurons"))

#titles = {'mC2L': 'Center-to-left', 'mC2R': 'Center-to-right',
#          'mL2C': 'Left-to-center', 'mR2C': 'Right-to-center'}
for label in ("mC2L", "mC2R", "mL2C", "mR2C","pC2L","pC2R"):
    decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath, label=label+"-")
    avgCorr = decodingMovementProgress.groupby(["shuffle","sess"]).apply(calcCorr).reset_index("shuffle")
    avgCorr["genotype"] = avgCorr.index.str.split("_").str[0]
    avgCorr.loc[avgCorr.shuffle, "genotype"] = "shuffled"
    avgCorr["animal"] = avgCorr.index.str.split("_").str[1]
    avgCorr.loc[avgCorr.shuffle, "animal"] = avgCorr.loc[avgCorr.shuffle, "animal"] + "_r"
    avgCorr["date"] = avgCorr.index.str.split("_").str[2]
    avgCorr.sort_values(["genotype", "animal", "date"], ascending=False, inplace=True)
    
    ax = layout.axes["movementProgressCorrelations_{}".format(label)]["axis"]
    sessionBarPlot.sessionBarPlot(avgCorr, yCol="correlation", weightCol="nNeurons",
                                  genotypeOrder=('d1','a2a','oprm1','shuffled'),
                                  ax=ax, colorFunc=style.getColor, weightScale=0.035)
    
    ax.axhline(0, ls=':', alpha=.5, lw=mpl.rcParams['axes.linewidth'],
               color='k', clip_on=False)
    ax.set_xticks(())
    ax.set_yticks((0,.5,1))
    ax.set_yticks((.25,.75), minor=True)
    if label=="mL2C":
        ax.set_ylabel("true X predicted\ncorrelation")
    else:
        ax.set_yticklabels([])

    ax.set_ylim(0,1)
    sns.despine(ax=ax, bottom=True)
    
axt = layout.axes['bar_legend']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.3,
                                 label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1',
                                        'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']
                  ]
axt.legend(handles=legend_elements, ncol=4, loc='center',
           mode='expand')
axt.axis('off')

#%% Panel D

examples = [('oprm1','5574',('190126', '190127', '190129', '190131', '190202'), 20),
            ('d1','5643',('190112', '190114', '190128', '190130', '190201'), 170),
            ('a2a','6043',('190114', '190126', '190128', '190130', '190201'), 287)]
alignmentStore = h5py.File(alignmentDataPath, "r")
def findAlignedNeuron(genotype, animal, fromDate, toDate, neuron):
    if fromDate == toDate:
        return neuron
    else:
        matches = alignmentStore["/data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)]
        return pd.Series(matches[:,1], matches[:,0]).loc[neuron]

saturation = 1
for i in range(3):
    for j in range(5):
        sess = next(readSessions.findSessions(endoDataPath, animal=examples[i][1],
                                             date=examples[i][2][j], task="2choice"))
        neuron = findAlignedNeuron(examples[i][0], examples[i][1], examples[i][2][0],
                                   examples[i][2][j], examples[i][3])
        signal = sess.readDeconvolvedTraces()[neuron]
        signal -= signal.mean()
        signal /= signal.std()
        ax = layout.axes["acrossDays_ex{}{}".format(i+1,j+1)]["axis"]
        fv = fancyViz.SchematicIntensityPlot(sess, splitReturns=True,
                                             linewidth=mpl.rcParams['axes.linewidth'],
                                             saturation=saturation, smoothing=7)
        img = fv.draw(signal, ax=ax)
    
    #axbg = layout.axes['acrossDays_ex{}1_bg'.format(i+1)]['axis']
    #axbg.axvspan(-.055, -.03, .1, .93, color=sel_colors[i], alpha=1,
    #             clip_on=False)
    #axbg.set_xlim((0,1))
    #axbg.set_axis_off()
alignmentStore.close()
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "decodingSupp.svg")