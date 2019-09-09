import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import h5py
import pathlib
import figurefirst

#import sys
#thisFolder = pathlib.Path(__file__).resolve().parent
#sys.path.append(str(thisFolder.parent))

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisForcedAlternation, analysisDecoding
import style

style.set_context()

endoDataPath = "endoData_2019.hdf"
alignmentDataPath = "alignment_190227.hdf"

outputFolder = pathlib.Path("svg")
cacheFolder = pathlib.Path("cache")
templateFolder = pathlib.Path(__file__).parent / "templates"

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "forcedAlternation.svg")
layout.make_mplfigures()

## Panel B
cachedDataPath = cacheFolder / "forcedAlternationTunings.pkl"
if cachedDataPath.is_file():
    tunings = pd.read_pickle(cachedDataPath)
else:
    tunings = analysisForcedAlternation.getTuningData(endoDataPath)
    tunings.to_pickle(cachedDataPath)

## Panel C
cachedDataPath = cacheFolder / "decodingAcrossDays.pkl"
if cachedDataPath.is_file():
    decodingAccrossDays = pd.read_pickle(cachedDataPath)
else:
    decodingAccrossDays = analysisDecoding.decodingAccrossDays(endoDataPath, alignmentDataPath)
    decodingAccrossDays.to_pickle(cachedDataPath)

def bootstrapSEM(values, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(values), len(values), replace=True)
        avgs.append(np.average(values.iloc[idx], weights=weights.iloc[idx]))
    return np.std(avgs)

fromDate = pd.to_datetime(decodingAccrossDays.fromDate, format="%y%m%d")
toDate = pd.to_datetime(decodingAccrossDays.toDate, format="%y%m%d")
td = (toDate - fromDate).dt.days
decodingAccrossDays["dayDifference"] = td

for i in range(2):
    if i == 0:
        selection = decodingAccrossDays.query("fromTask=='2choice' & toTask=='forcedAlternation' & dayDifference <= 7")
        plt.sca(layout.axes["decoding2choiceToFA"]["axis"])
    else:
        selection = decodingAccrossDays.query("fromTask=='forcedAlternation' & toTask=='2choiceAgain'")
        plt.sca(layout.axes["decodingFATo2choice"]["axis"])
    g = selection.groupby(["animal", "fromDate", "toDate"])

    perAnimal = g.mean()[['nNeurons', 'sameDayScore', 'nextDayScore', 'sameDayShuffled', 'nextDayShuffled']]
    perAnimal["genotype"] = g.genotype.first()

    scaledScore = perAnimal[['sameDayScore', 'nextDayScore']] * perAnimal.nNeurons[:,np.newaxis]
    perGenotype = scaledScore.groupby(perAnimal.genotype).sum()
    perGenotype /= perAnimal.groupby("genotype").nNeurons.sum()[:, np.newaxis]

    shuffleScore = perAnimal[['sameDayShuffled', 'nextDayShuffled']] * perAnimal.nNeurons[:,np.newaxis]
    shuffleScore = shuffleScore.sum(axis=0) / perAnimal.nNeurons.sum()
    
    for r in perAnimal.itertuples():
        plt.plot([0,1], [r.sameDayScore, r.nextDayScore], lw=style.lw()*r.nNeurons/400.0,
                 c=style.getColor(r.genotype), alpha=0.2)
    for r in perGenotype.itertuples():
        gt = r.Index
        animalsWithGt = perAnimal.query("genotype == '{}'".format(gt))
        sameDaySEM = bootstrapSEM(animalsWithGt.sameDayScore, animalsWithGt.nNeurons)
        nextDaySEM = bootstrapSEM(animalsWithGt.nextDayScore, animalsWithGt.nNeurons)
        plt.errorbar([0,1], [r.sameDayScore, r.nextDayScore], [sameDaySEM, nextDaySEM],
                     lw=style.lw(), c=style.getColor(gt))
    plt.plot([0,1], [shuffleScore.sameDayShuffled, shuffleScore.nextDayShuffled],
             lw=style.lw(), c=style.getColor("shuffled"))
    plt.ylim(0,1)
    plt.xlim(-0.25, 1.25)
    if i == 0:
        plt.xticks((0,1), ("2-choice", "Forced\nalternation"))
        plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
        plt.ylabel("Decoding accuracy (%)")
    else:
        plt.xticks((0,1), ("Forced\nalternation", "2-choice"))
        plt.yticks(np.linspace(0,1,5), [""]*5)
    sns.despine(ax=plt.gca())

## Panel D
alignmentStore = h5py.File(alignmentDataPath, "r")
def findAlignedNeuron(genotype, animal, fromDate, toDate, neuron):
    if fromDate == toDate:
        return neuron
    else:
        matches = alignmentStore["/data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)]
        return pd.Series(matches[:,1], matches[:,0]).loc[neuron]

examples = [("oprm1", "5703", '190114', ('190201', '190204', '190206', '190207', '190208'), 112),
            ("d1", "5643", '190112', ('190201', '190203', '190205', '190206', '190207'), 170),
            ("a2a", "5693", '190115', ('190202', '190204', '190206', '190207', '190208'), 67)]
for i in range(3):
    for j in range(5):
        sess = next(readSessions.findSessions(endoDataPath, animal=examples[i][1], date=examples[i][3][j],
                                              task=["2choice", "forcedAlternation", "2choiceAgain"]))
        neuron = findAlignedNeuron(examples[i][0], examples[i][1], examples[i][2],
                                   examples[i][3][j], examples[i][4])
        signal = sess.readDeconvolvedTraces()[neuron]
        signal -= signal.mean()
        signal /= signal.std()
        ax = layout.axes["accrossDays_ex{}{}".format(i+1,j+1)]["axis"]
        fv = fancyViz.SchematicIntensityPlot(sess, linewidth=style.lw()*0.5)
        fv.draw(signal, ax=ax)

layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "forcedAlternation.svg")