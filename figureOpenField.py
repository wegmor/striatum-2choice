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

import analysisOpenField
import style

style.set_context()
plt.ioff()

#%%
endoDataPath = pathlib.Path('.') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('.') / "alignment_190227.hdf"
outputFolder = pathlib.Path('svg')
cacheFolder = pathlib.Path('cache')
templateFolder = pathlib.Path('striatum_2choice/templates')

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "openField.svg")
layout.make_mplfigures()

#All 4-behavior panels    
cachedDataPath = cacheFolder / "segmentedBehavior.pkl"
if cachedDataPath.is_file():
    segmentedBehavior = pd.read_pickle(cachedDataPath)
else:
    segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
    segmentedBehavior.to_pickle(cachedDataPath)

segmentedBehavior = segmentedBehavior.set_index("session")


## Panel A
cachedDataPath = cacheFolder / "openFieldTunings.pkl"
if cachedDataPath.is_file():
    tuningData = pd.read_pickle(cachedDataPath)
else:
    tuningData = analysisOpenField.getTuningData(endoDataPath, segmentedBehavior)
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
        ax.set_xticklabels(order, rotation=45, ha="right")
        title = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g]
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
    hm = sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, annot=True, fmt=".2f",
                     cmap=cmocean.cm.balance, cbar=True, cbar_ax=cax,
                     cbar_kws={'ticks':(-1,0,1), 'orientation': 'horizontal'},
                     annot_kws={'fontsize': 4.0}, yticklabels=(genotype=="oprm1"),
                     linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[genotype])


#Panel C
cachedDataPath = cacheFolder / "openFieldDecodingWithIncreasingNumberOfNeurons.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisOpenField.decodeWithIncreasingNumberOfNeurons(endoDataPath, segmentedBehavior)
    decodingData.to_pickle(cachedDataPath)

decodingData.insert(1, "genotype", decodingData.session.str.split("_").str[0])
plt.sca(layout.axes["decodeWithIncreasingNumberOfNeurons"]["axis"])
for strSess, df in decodingData.groupby("session"):
    genotype = strSess.split("_")[0]
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=0.2, lw=.5)
    plt.plot(df.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
             alpha=0.2, lw=.5)
for genotype, df in decodingData.groupby("genotype"):
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=1.0)
plt.plot(decodingData.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
         alpha=1.0)

order = ("oprm1", "d1", "a2a")
meanHandles = [mpl.lines.Line2D([], [], color=style.getColor(g)) for g in order]
shuffleHandle = mpl.lines.Line2D([], [], color=style.getColor("shuffled"))
plt.legend(meanHandles+[shuffleHandle], order+("shuffled",), loc=(0.45, 0.45), ncol=2)

plt.ylim(0,1)
plt.xlim(0,300)
plt.xlabel("Number of neurons")
plt.ylabel("Decoding accuracy (%)")
plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
sns.despine(ax=plt.gca())

## Panel D
cachedDataPath = cacheFolder / "openFieldDecodingConfusion.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisOpenField.decodingConfusion(endoDataPath, segmentedBehavior)
    decodingData.to_pickle(cachedDataPath)
order = ["stationary", "running", "leftTurn", "rightTurn"]
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]
for gt, data in decodingData.groupby("genotype"):
    weightedData = data.set_index(["true", "predicted"]).eval("occurencies * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    ax = layout.axes["confusionMatrix_{}".format(gt)]["axis"]
    yticks = order if gt == "oprm1" else False
    sns.heatmap(weightedData[order].reindex(order), ax=ax, vmin=0, vmax=1, annot=True, fmt=".0%", cmap=cmocean.cm.amp,
                cbar=False, xticklabels=order, yticklabels=yticks, annot_kws={'fontsize': 4.5},
                linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel("Predicted" if gt=="d1" else None)
    ax.set_ylabel("Truth" if gt=="oprm1" else None)
    ax.set_title(gt[0].upper() + gt[1:])

## Panel E
cachedDataPath = cacheFolder / "actionTunings.pkl"
if cachedDataPath.is_file():
    twoChoiceTunings = pd.read_pickle(cachedDataPath)
else:
    twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)
    twoChoiceTunings.to_pickle(cachedDataPath)

twoChoiceTunings = twoChoiceTunings.set_index(["animal", "date", "neuron"])[["action", "pct", "tuning"]]
joinedTunings = tuningData.join(twoChoiceTunings, on=["animal", "date", "neuron"], rsuffix="_2choice")
corrMeasure = lambda df: scipy.stats.pearsonr(df.tuning, df.tuning_2choice)[0]
correlations = joinedTunings.groupby(["genotype", "action", "action_2choice"]).apply(corrMeasure).unstack()

cax = layout.axes['corr_of2c_colorbar']['axis']
cax.tick_params(axis='y', which='both',length=0)

order2choice = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co", "pL2Cr",
                "pC2L-", "pC2R-", "pR2Cd", "pR2Co", "pR2Cr"]
for gt, perGt in correlations[order2choice].groupby(level=0):
    ax = layout.axes["openField2choiceCorrs_{}".format(gt)]["axis"]
    sns.heatmap(perGt.loc[gt].loc[order], ax=ax, vmin=-1, vmax=1, annot=True,
                fmt=".2f", cmap=cmocean.cm.balance, cbar=True, cbar_ax=cax,
                cbar_kws={'ticks':(-1,0,1)}, xticklabels=False,
                annot_kws={'fontsize': 4.0}, yticklabels=order,
                linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(gt[0].upper() + gt[1:])
    ax.set_ylim(4,0)
layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "openField.svg")
