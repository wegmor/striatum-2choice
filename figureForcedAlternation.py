import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FixedLocator
import h5py
import pathlib
import figurefirst
from collections import defaultdict

#import sys
#thisFolder = pathlib.Path(__file__).resolve().parent
#sys.path.append(str(thisFolder.parent))

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisForcedAlternation, analysisDecoding, analysisTunings
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
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisForcedAlternation.getTuningData(endoDataPath)
    tuningData.to_pickle(cachedDataPath)
    
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005
df = tuningData.copy()

# only keep max tuning for each neuron
maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~df.signp, 'action'] = 'none' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

# create dictionary with modified alpha to separate r/o/d phases
cdict = defaultdict(lambda: np.array([1,1,1]),
                    {a:style.getColor(a[:4]) for a 
                     in ['mC2L-','mC2R-','mL2C-','mR2C-','pC2L-','pC2R-','pL2C-','pR2C-']})
cdict['pL2Cr'] = cdict['pL2C-']
cdict['pL2Co'] = np.append(cdict['pL2C-'], .45)
cdict['pL2Cd'] = np.append(cdict['pL2C-'], .7)
cdict['pR2Cr'] = cdict['pR2C-']
cdict['pR2Co'] = np.append(cdict['pR2C-'], .45)
cdict['pR2Cd'] = np.append(cdict['pR2C-'], .7)
cdict['pC2L-'] = np.append(cdict['pC2L-'], .45)

for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']

    gdata = maxdf.loc[g]   
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[cdict[a] for a in gdata.index])

    ax.set_aspect('equal')

#%% tuning counts (simple)
hist_df = analysisTunings.getTunedNoHistData(tuningData)

axs = {}
for g, gdata in hist_df.query('bin != 0').groupby('genotype'):
    ax = layout.axes['no_tuned_'+g]['axis']
    axs[g] = ax
    
    ax.scatter(analysisTunings.jitter(gdata.bin, .12), gdata.signp,
               s=gdata.noNeurons/25, edgecolor=style.getColor(g),
               facecolor='none', alpha=.8, zorder=0, clip_on=False,
               lw=mpl.rcParams['axes.linewidth'])
    
    avg = gdata.groupby('bin').apply(analysisTunings.wAvg, 'signp', 'noNeurons')
    sem = gdata.groupby('bin').apply(analysisTunings.bootstrap, 'signp', 'noNeurons')
    ax.bar(avg.index, avg, yerr=sem, color=style.getColor(g),
           lw=0, alpha=.3, zorder=1)
    
    ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])
    ax.set_ylim((0,.5))
    ax.set_yticks((0,.25,.5))
    ax.set_yticklabels(())
    ax.yaxis.set_minor_locator(MultipleLocator(.125))
    ax.set_xlim((0.25,5.75))
    ax.set_xticks((1,3,5))
    ax.xaxis.set_minor_locator(FixedLocator((2,4)))
    ax.set_xticklabels(['1','3','5+'])
    sns.despine(ax=ax)

axs['d1'].set_yticklabels((0,25,50))
axs['d1'].set_ylabel('neurons (%)')
axs['a2a'].set_xlabel('number of actions')
    
    
## Panel D
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

## Panel E
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