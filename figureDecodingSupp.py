import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import pathlib
import figurefirst
import cmocean

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisDecoding
import style

style.set_context()
plt.ioff

#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")
if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "decodingSupp.svg")
layout.make_mplfigures()

#%% Panel A
decodingData = analysisDecoding.decodingConfusion(endoDataPath)

order = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "dL2C-", "pL2Co", "pL2Cr",
         "pC2L-", "pC2R-", "dR2C-", "pR2Co", "pR2Cr"]
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
cax.text(.5, 1.1, 'recall (%)', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)

#%% Panel ?
decodingData = analysisDecoding.decodeWithIncreasingNumberOfNeurons(endoDataPath)
for nNeurons in (100, 175, "all"):
    if nNeurons == "all":
        maxNeurons = decodingData.groupby("session").nNeurons.max()
        acc = decodingData.join(maxNeurons.rename("maxNeurons"), on="session")
        acc = acc[acc.nNeurons == acc.maxNeurons]
    else:
        acc = decodingData[decodingData.nNeurons==nNeurons]
    acc = acc.groupby("session").mean().reset_index()
    genotype, animal, date = acc.session.str.split("_").str
    acc["genotype"] = genotype
    acc["animal"] = animal
    acc["date"] = date

    cols = ["genotype", "animal", "date", "nNeurons", "accuracy"]
    real = acc.rename(columns={'realAccuracy': 'accuracy'})[cols]
    shuffled = acc.rename(columns={'shuffledAccuracy': 'accuracy'})[cols]
    shuffled["genotype"] = "shuffled"
    shuffled["animal"] += "_shuffled"
    acc = pd.concat((real, shuffled))
    acc["accuracy"] *= 100
    if nNeurons == "all":
        ax = layout.axes['decodingAll']['axis']
        ax.set_title("all recorded neurons")
    else:
        ax = layout.axes['decodingAt{}'.format(nNeurons)]['axis']
        ax.set_title("subsampling {} neurons".format(nNeurons))
    sessionBarPlot.sessionBarPlot(acc, yCol="accuracy", weightCol="nNeurons",
                                  genotypeOrder=('d1','a2a','oprm1','shuffled'),
                                  ax=ax, colorFunc=style.getColor, weightScale=0.035, xlabels="animal")
    sns.despine(ax=ax)
    ax.set_ylim(0,100)
    if nNeurons == 100:
        ax.set_ylabel("decoding accuracy (%)")
        ax.text(3, -7, "(same animals\nshuffled)", fontsize=6, ha="center",
                va="top", clip_on=False)
        ax.set_xlabel("animal ID", labelpad=6)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("animal", labelpad=6)
    perGt = acc[~acc.animal.str.endswith("_shuffled")]
    perGt = perGt.groupby(["genotype", "animal"]).accuracy.median()
    print("nNeurons = {}".format(nNeurons))
    print("N=", perGt.groupby(level=0).size())
    print('\t', scipy.stats.kruskal(*[d[1].values for d in perGt.groupby(level=0)]))

axt = layout.axes['decoding_legend_genotype']['axis']
legend_elements = [mpatches.Patch(color=style.getColor(gt), alpha=.3,
                                  label={'oprm1':'Oprm1', 'a2a':'A2A', 'd1':'D1',
                                        'shuffled':'shuffled'}[gt])
                   for gt in ['d1','a2a','oprm1','shuffled']]
axt.legend(handles=legend_elements, ncol=2, loc='center', mode='expand')
axt.axis('off')
axt = layout.axes['decoding_legend_nNeurons']['axis']
legend_elements = [mpl.lines.Line2D([], [], marker='o', linestyle="none", markeredgecolor='k',
                                    markerfacecolor="none", markersize=np.sqrt(0.035*i),
                                    markeredgewidth=0.5, label=str(i))
                   for i in (25, 50, 100, 300, 600)]
axt.legend(handles=legend_elements, loc='upper left', 
           mode="expand", title="num. neurons")#, mode='expand')
axt.axis('off')

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
        ax.set_ylabel("r(true, predicted)")
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

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "decodingSupp.svg")