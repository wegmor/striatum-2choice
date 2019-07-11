import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import figurefirst

import sys
thisFolder = pathlib.Path(__file__).resolve().parent
sys.path.append(str(thisFolder.parent))

import fancyViz
import analysisDecoding
import style

style.set_context("paper")

endoDataPath = "endoData_2019.hdf"
outputFolder = thisFolder / "svg"
if not outputFolder.is_dir():
    outputFolder.mkdir()

layout = figurefirst.FigureLayout(thisFolder / "templates" / "decoding.svg")
layout.make_mplfigures()

## Panel A
cachedDataPath = thisFolder / "cache" / "decodeWithIncreasingNumberOfNeurons.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisDecoding.decodeWithIncreasingNumberOfNeurons(endoDataPath)
    if not cachedDataPath.parent.is_dir():
        cachedDataPath.parent.mkdir()
    decodingData.to_pickle(cachedDataPath)
decodingData.insert(1, "genotype", decodingData.session.str.split("_").str[0])
plt.sca(layout.axes["decodeWithIncreasingNumberOfNeurons"]["axis"])
for strSess, df in decodingData.groupby("session"):
    genotype = strSess.split("_")[0]
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype), alpha=0.2, lw=style.lw()*0.5)
    plt.plot(df.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"), alpha=0.2, lw=style.lw()*0.5)
for genotype, df in decodingData.groupby("genotype"):
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype), alpha=1.0, lw=style.lw()*2)
plt.plot(decodingData.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"), alpha=1.0, lw=style.lw()*2)
order = ("oprm1", "d1", "a2a")
#meanHandles = [matplotlib.lines.Line2D([], [], color=style.getColor(g), lw=style.lw()*2) for g in order]
#shuffleHandle = matplotlib.lines.Line2D([], [], color=style.getColor("shuffled"), lw=style.lw()*2)
#plt.legend(meanHandles+[shuffleHandle], order+("shuffled",), loc=(1.02, 0.18))
plt.ylim(0,1)
plt.xlim(0, 200)
plt.xlabel("Number of neurons")
plt.ylabel("Decoding accuracy (%)")
plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
sns.despine()

##Panel B
cachedDataPath = thisFolder / "cache" / "decodingConfusionDiagonal.pkl"
if cachedDataPath.is_file():
    confusionDiagonal = pd.read_pickle(cachedDataPath)
else:
    confusionDiagonal = analysisDecoding.decodingConfusionDiagonal(endoDataPath)
    if not cachedDataPath.parent.is_dir():
        cachedDataPath.parent.mkdir()
    confusionDiagonal.to_pickle(cachedDataPath)
means = confusionDiagonal.groupby("sess").mean()
nNeurons = means.nNeurons
labels = list(means.columns)
for i in range(6):
     labels[i] = labels[i][:4]
genotypes = means.index.str.split("_").str[0]
for i, gt in enumerate(("oprm1", "d1", "a2a")):
    gtMeans = np.average(means[genotypes==gt].drop("nNeurons", axis=1), axis=0, weights=nNeurons[genotypes==gt])

    cmap = {"oprm1": plt.cm.Greens, "d1": plt.cm.Reds, "a2a": plt.cm.Blues}[gt]
    di = {k: cmap(v) for k, v in zip(labels, gtMeans)}
    plt.sca(layout.axes["decodingAccuracyPerLabel_{}".format(gt)]["axis"])
    lw = matplotlib.rcParams["lines.linewidth"]
    fancyViz.drawBinnedSchematicPlot(di, lw=lw)
   
    cb1 = matplotlib.colorbar.ColorbarBase(layout.axes["decodingAccuracyCbar_{}".format(gt)], cmap=cmap,
                                           norm=matplotlib.colors.Normalize(vmin=0, vmax=100),
                                           orientation='vertical')
    
##Panel C
cachedDataPath = thisFolder / "cache" / "decodingAccrossDays.pkl"
if cachedDataPath.is_file():
    decodingAccrossDays = pd.read_pickle(cachedDataPath)
else:
    decodingAccrossDays = analysisDecoding.decodingAccrossDays(endoDataPath)
    if not cachedDataPath.parent.is_dir():
        cachedDataPath.parent.mkdir()
    decodingAccrossDays.to_pickle(cachedDataPath)

#accrossDays = accrossDays.rename(columns={"sameDayShuffled": "nextDayScore", "nextDayScore": "sameDayShuffled"})
fromDate = pd.to_datetime(decodingAccrossDays.fromDate, format="%y%m%d")
toDate = pd.to_datetime(decodingAccrossDays.toDate, format="%y%m%d")
td = (toDate - fromDate).dt.days
decodingAccrossDays["dayDifference"] = td

selection = decodingAccrossDays.query("fromTask=='2choice' & toTask=='2choice'")
for i,l,h in ((0,1,3), (1,4,6), (2,7,14), (3,14,100)):
    g = selection.query("dayDifference >= {} & dayDifference <= {}".format(l,h)).groupby(["animal", "fromDate", "toDate"])
    nNeurons = g.nNeurons.mean()
    sameDayMeans = g.sameDayScore.mean()
    sameDaySDs = g.sameDayScore.std()
    nextDayMeans = g.nextDayScore.mean()
    nextDaySDs = g.nextDayScore.std()
    sameDayShuffled = g.sameDayShuffled.mean()
    nextDayShuffled = g.sameDayShuffled.mean()
    
    both = pd.concat([sameDayMeans, nextDayMeans], axis=1)
    bothShuffled = pd.concat([sameDayShuffled, nextDayShuffled], axis=1)

    plt.sca(layout.axes["decodingAccrossDays_{}".format(i+1)]["axis"])
    colors = [style.getColor(gt) for gt in g.genotype.first()]
    plt.scatter(np.zeros(len(both)), sameDayShuffled.values, s=nNeurons/5,
                edgecolor=style.getColor("shuffled"), marker="o", facecolor="none", lw=style.lw())
    plt.scatter(np.ones(len(both)), nextDayShuffled.values, s=nNeurons/5,
                edgecolor=style.getColor("shuffled"), marker="o", facecolor="none", lw=style.lw())
    plt.scatter(np.zeros(len(both)), sameDayMeans.values, s=nNeurons/5,
                edgecolor=colors, marker="o", facecolor="none", lw=style.lw())
    plt.scatter(np.ones(len(both)), nextDayMeans.values, s=nNeurons/5,
                edgecolor=colors, marker="o", facecolor="none", lw=style.lw())
    for gt, b in both.groupby(g.genotype.first()):
        plt.plot(b.T.values, color=style.getColor(gt), lw=style.lw(), alpha=0.5)
    plt.plot(bothShuffled.T.values.mean(axis=1), color=style.getColor("shuffled"), lw=3)
    plt.ylim(0,1)
    plt.xlim(-0.4, 1.4)
    xlab = ("1-3 days\nlater", "4-6 days\nlater", "7-14 days\nlater", "14+ days\nlater")
    plt.xticks((0,1), ("Same\nday", xlab[i]))
    if i==0:
        plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
        plt.ylabel("Decoding accuracy (%)")
    else:
        plt.yticks(np.linspace(0,1,5), [""]*5)
    #plt.xlabel("Day")
    sns.despine(ax=plt.gca())

    
layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "decoding.svg")
#plt.savefig(outputFolder.join("decoding.svg"), dpi=300, bbox_inches="tight")