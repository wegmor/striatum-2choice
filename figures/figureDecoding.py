import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import h5py
import pathlib
import figurefirst

import sys
thisFolder = pathlib.Path(__file__).resolve().parent
sys.path.append(str(thisFolder.parent))

import fancyViz
import readSessions
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
sns.despine(ax=plt.gca())

## Panel B
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

## Panel C
alignmentStore = h5py.File("alignment_190227.hdf", "r")
def findAlignedNeuron(genotype, animal, fromDate, toDate, neuron):
    if fromDate == toDate:
        return neuron
    else:
        matches = alignmentStore["/data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)]
        return pd.Series(matches[:,1], matches[:,0]).loc[neuron]

examples = [("oprm1", "5703", ('190114', '190116', '190201'), 16),
            ("d1", "5643", ('190112', '190114', '190128'), 67),
            ("a2a", "6043", ('190114', '190126', '190128'), 63)]
for i in range(3):
    for j in range(3):
        sess = next(readSessions.findSessions(endoDataPath, animal=examples[i][1],
                                             date=examples[i][2][j], task="2choice"))
        neuron = findAlignedNeuron(examples[i][0], examples[i][1], examples[i][2][0],
                                   examples[i][2][j], examples[i][3])
        signal = sess.readDeconvolvedTraces()[neuron]
        signal -= signal.mean()
        signal /= signal.std()
        ax = layout.axes["accrossDays_ex{}{}".format(i+1,j+1)]["axis"]
        fv = fancyViz.SchematicIntensityPlot(sess, linewidth=style.lw()*0.5)
        fv.draw(signal, ax=ax)

## Panel D
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
    sns.despine(ax=plt.gca())

## Panel E
sess = next(readSessions.findSessions("endoData_2019.hdf", animal="5308", date="190131"))
lfa = sess.labelFrameActions(reward="sidePorts")
deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
X = deconv[lfa.label=="mR2C-"]
Y = lfa.actionProgress[lfa.label=="mR2C-"]
avgActivity = X.groupby((Y*10).astype("int")/10.0).mean().T
sorting = avgActivity.idxmax(axis=1).argsort()
plt.sca(layout.axes["movementProgressRaster"]["axis"])
plt.imshow(avgActivity.iloc[sorting], aspect="auto",
           interpolation="nearest", vmin=-1, vmax=1, cmap="RdYlBu_r")
plt.xticks((-0.5,4.5,9.5), (0, 50, 100))
plt.yticks([0, len(sorting)-1], [len(sorting), 0])
plt.xlabel("Progress (%)")
plt.ylabel("Neuron (by peak)")

exampleNeurons = (7, 66, 13)
fv = fancyViz.SchematicIntensityPlot(sess, linewidth=style.lw()*0.5,
                                     splitReturns=False, smoothing=7)
for i in range(3):
    ax = layout.axes["movementExample{}".format(i+1)]["axis"]
    fv.draw(deconv[exampleNeurons[i]], ax=ax)
    
## Panel F
cachedDataPath = thisFolder / "cache" / "decodingMovementProgress.pkl"
if cachedDataPath.is_file():
    decodingMovementProgress = pd.read_pickle(cachedDataPath)
else:
    decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath)
    if not cachedDataPath.parent.is_dir():
        cachedDataPath.parent.mkdir()
    decodingMovementProgress.to_pickle(cachedDataPath)
    
def calcCorr(df):
    r = scipy.stats.pearsonr(df.true, df.predicted)[0]
    return pd.Series((r, df.nNeurons.iloc[0]), ("correlation", "nNeurons"))

exampleSession = decodingMovementProgress.query("sess == 'oprm1_5308_190131' & not shuffle")
means = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.mean()
stds = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.std()
plt.sca(layout.axes["decodingProgressExample"]["axis"])
plt.plot([0,100], [0, 100], 'k--', alpha=0.2)
plt.errorbar(means.index*100, means*100, yerr=stds*100, fmt='.-', ms=10, color=style.getColor("oprm1"))
plt.xlim(-5,100)
plt.ylim(-5,100)
plt.yticks((0,50,100))
plt.xlabel("True progress (%)")
plt.ylabel("Predicted progress (%)")
corr = calcCorr(exampleSession).loc["correlation"]
plt.text(100, 10, "r = {:.3f}".format(corr), fontsize=8, color="k", ha="right")
sns.despine(ax=plt.gca())


## Panel G
avgCorr = decodingMovementProgress.query("not shuffle").groupby("sess").apply(calcCorr)
avgCorr["genotype"] = avgCorr.index.str.split("_").str[0]
avgCorr["animal"] = avgCorr.index.str.split("_").str[1]
avgCorr["date"] = avgCorr.index.str.split("_").str[2]
avgCorr.sort_values(["genotype", "animal", "date"], ascending=False, inplace=True)
#shuffledCorr = calcCorr(decodingMovementProgress.query("shuffle").set_index("sess")).to_frame()
#shuffledCorr["genotype"] = "shuffled"
#shuffledCorr["animal"] = shuffledCorr.index.str.split("_").str[1]
#shuffledCorr["date"] = shuffledCorr.index.str.split("_").str[2]
#avgCorr = pd.concat((avgCorr, shuffledCorr))
x = 7*avgCorr.groupby("genotype", sort=False).ngroup() + avgCorr.groupby("animal", sort=False).ngroup()
colors = [style.getColor(gt) for gt in avgCorr.genotype]
plt.sca(layout.axes["movementProgressCorrelations"]["axis"])
xticks = x.groupby(avgCorr.genotype).mean()
genotypeMeans = avgCorr.groupby("genotype").apply(lambda df: np.average(df.correlation, weights=df.nNeurons))
barColors = [style.getColor(gt) for gt in genotypeMeans.index]
barColorsAlpha = [matplotlib.colors.to_rgb(c)+(0.1,) for c in barColors]
barWidths = avgCorr.groupby("genotype").animal.nunique().reindex(genotypeMeans.index) + 3
plt.bar(xticks, genotypeMeans, barWidths, color=barColorsAlpha, edgecolor=barColors, linewidth=style.lw())
plt.scatter(x, avgCorr.correlation, avgCorr.nNeurons/20, colors, alpha=0.5)
plt.xticks(xticks, xticks.index)
plt.ylim(0,1)
sns.despine(ax=plt.gca())
plt.ylabel("Correlation")
    
layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "decoding.svg")
#plt.savefig(outputFolder.join("decoding.svg"), dpi=300, bbox_inches="tight")