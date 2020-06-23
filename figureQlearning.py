import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import figurefirst
import style
import analysisStaySwitchDecoding
import cmocean
import scipy.stats
from utils import readSessions, sessionBarPlot, fancyViz
plt.ioff()

#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "qLearning.svg")
layout.make_mplfigures()

#Copied from figure5. TODO: This should be written much briefer
cachedDataPaths = [cacheFolder / name for name in ['logRegCoefficients.pkl',
                                                   'logRegDF.pkl']]
if np.all([path.is_file() for path in cachedDataPaths]):
    logRegCoef = pd.read_pickle(cachedDataPaths[0])
    logRegDF = pd.read_pickle(cachedDataPaths[1])
else:
    logRegCoef, logRegDF = analysisStaySwitchDecoding.getAVCoefficients(endoDataPath)
    logRegCoef.to_pickle(cachedDataPaths[0])
    logRegDF.to_pickle(cachedDataPaths[1])
cachedDataPath = cacheFolder / 'actionValues.pkl'
if cachedDataPath.is_file():
    regAVs = pd.read_pickle(cachedDataPath)
else:
    regAVs = analysisStaySwitchDecoding.getActionValues(endoDataPath, logRegCoef)
    regAVs.to_pickle(cachedDataPath)
regAVs["session"] = regAVs.genotype + "_" + regAVs.animal + "_" + regAVs.date
regAVs = regAVs.set_index(["session", "actionNo"])[["label", "value"]]
    
qAVs = analysisStaySwitchDecoding.getQActionValues(endoDataPath)
qAVs = qAVs.drop("oprm1_3323_180327", axis=0)
regAVs = regAVs.drop("oprm1_3323_180327", axis=0)

ax = layout.axes['ex_trace']['axis']
exSess = next(readSessions.findSessions(endoDataPath, animal="5703", date="190130"))
lfa = exSess.labelFrameActions()
lfa = lfa.join(qAVs.loc[str(exSess)][["Q_actionValue"]], on="actionNo")
lfa = lfa.join(regAVs.loc[str(exSess)][["value"]], on="actionNo")
ax.plot(lfa.Q_actionValue, label="Q action value")
ax.plot(lfa.value, label="regression action value")
for port in ("pC", "pL", "pR"):
    ax.fill_between(lfa.index, -6, 6, lfa.label.str.startswith(port), alpha=.25, color=style.getColor(port))
ax.set_xlim(10000, 20000)
ax.set_ylim(-7, 7)
ax.set_xticks([])
rewards = lfa.loc[lfa.label.str.endswith('r').astype('int').diff()==1].index.values
for r in rewards:
    fancyViz.drawWaterDrop(ax, np.array([r, 6]), np.array([75, 1]),
                           facecolor='k')
    ax.axvline(r, .05, .85, lw=.5, ls='--', color='k')
ax.hlines(-7, 18800, 20000, clip_on=False, color='k')
ax.text(19400, -8, "1 min".format(1000/20), ha="center", va="top", fontsize=6)
sns.despine(ax=ax, bottom=True)
ax.legend(ncol=2, bbox_to_anchor=(0.55, 1.05, 0.45, 0.1), mode="expand")

q_choices = qAVs[qAVs.label.isin(("pC2L-", "pC2R-"))].copy()
q_choices["rightChoice"] = q_choices.label=="pC2R-"

bins = np.arange(-1,1.1,0.1)
binned = pd.cut(q_choices.Qr_minus_Ql, bins)
fracRight = q_choices.groupby(binned).rightChoice.mean()
xx = (bins[1:] + bins[:-1])/2
ax = layout.axes['diffQ']['axis']
ax.plot(xx, 100*fracRight, ".-")
ax.set_xlabel("$Q_R - Q_L$")
ax.set_ylabel("right choice (%)")
ax.set_ylim(0,100)
sns.despine(ax=ax)

bins = np.arange(-5.5,5.6)
binned = pd.cut(q_choices.Q_actionValue, bins)
fracRight = q_choices.groupby(binned).rightChoice.mean()
xx = (bins[1:] + bins[:-1])/2
ax = layout.axes['Qav']['axis']
ax.plot(xx, 100*fracRight, ".-")
ax.set_xlabel("Q action value")
ax.set_yticklabels([])
ax.set_ylim(0,100)
#ax.set_ylabel("right choice (%)")
sns.despine(ax=ax)

params = analysisStaySwitchDecoding.fitQParameters(endoDataPath)
yticks = {'alpha': [0, 0.25, 0.5], 'beta': [0, 5, 10, 15],
          'bias': [-2, -1, 0, 1, 2]}
for p in ("alpha", "beta", "bias"):
    ax = layout.axes['params_'+p]['axis']
    sns.swarmplot(y=params[p], ax=ax, size=1.5)
    ax.set_ylabel("")
    ax.set_yticks(yticks[p])
    ax.set_xticks([])
    ax.set_ylim(yticks[p][0], yticks[p][-1])
    ax.set_title(p)
    sns.despine(ax=ax, bottom=True)

ax = layout.axes['Qav_vs_regav']['axis']
mask = qAVs.label.isin(("pC2L-", "pC2R-"))
ax.plot(qAVs.Q_actionValue[mask], regAVs.value[mask], 'k.', markersize=.2, alpha=.5, rasterized=True)
#Figurefirst axis has proportions 3:4
ax.set_xlim(-10, 10)
ax.set_ylim(-7.5, 7.5)
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-5, 0, 5])
ax.set_xlabel("Q action value")
ax.set_ylabel("regression action value")
sns.despine(ax=ax)

layout.insert_figures('plots')
layout.write_svg(outputFolder / "qLearning.svg")