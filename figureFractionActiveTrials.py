import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pathlib
import figurefirst
import itertools

import analysisFractionActiveTrials, analysisTunings
from utils import readSessions
import style

style.set_context()
plt.ioff()

#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path('svg')
templateFolder = pathlib.Path('templates')

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "fractionActiveTrials.svg")
layout.make_mplfigures()

tunings = analysisTunings.getTuningData(endoDataPath).astype({'neuron': np.int})
posTuned = tunings.set_index(["animal", "date", "action", "neuron"]).pct > 0.995
posTuned = posTuned.rename("posTuned").sort_index()

fracActive = analysisFractionActiveTrials.getFractionActiveTrials(endoDataPath)
split = fracActive.sess.str.split("_").str
fracActive["genotype"] = split[0]
fracActive["animal"] = split[1]
fracActive["date"] = split[2]
fracActive = fracActive.join(posTuned, on=["animal", "date", "label", "neuron"], how="inner")
fracActive = fracActive.set_index(["genotype", "label"]).sort_index()

labels = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "pL2Cd", "pL2Co",
          "pL2Cr", "pC2L-", "pC2R-", "pR2Cd", "pR2Co","pR2Cr"]
for gt, l in itertools.product(("d1", "a2a", "oprm1"), labels):
    if l[-1] == '-': axName = l[:4]+"_"+gt
    else: axName = l+"_"+gt

    subset = fracActive.loc[(gt, l)]
    topbin_x = 0
    topbin_y = 0
    for tuned in (False, True):
        color = style.getColor(l[:4]) if tuned else "gray"
        data = subset[subset.posTuned == tuned]
        
        ax = layout.axes[axName]['axis']
        ax.plot(data.fracTrialsActive*100, data.avgActivity,
                '.', color=color, ms=1, rasterized=True,
                markeredgewidth=0)
        
        ax = layout.axes[axName+"_x"]['axis']
        h = ax.hist(data.fracTrialsActive*100, bins=np.arange(0,101,2.5),
                    color=color, histtype="stepfilled")
        topbin_x = max(topbin_x, np.max(h[0]))
        
        ax = layout.axes[axName+"_y"]['axis']
        h = ax.hist(data.avgActivity, bins=np.arange(0,16,0.25), color=color, orientation="horizontal", histtype="stepfilled")
        topbin_y = max(topbin_y, np.max(h[0]))
        
    ax = layout.axes[axName]['axis']
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.set_xticks((0,50,100))
    ax.set_yticks((0,5,10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine(ax=ax)

    ax = layout.axes[axName+"_x"]['axis']
    ax.axhline(0, color='k', lw=.25, alpha=1)
    ax.set_xlim(0,100)
    ax.set_ylim(-0.2*topbin_x, 1.1*topbin_x)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    ax = layout.axes[axName+"_y"]['axis']
    ax.axvline(0, color='k', lw=.25, alpha=1)
    ax.set_ylim(0, 10)
    ax.set_xlim(-0.2*topbin_y, 1.1*topbin_y)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
ax = layout.axes["pL2Cr_d1"]['axis']
ax.set_xticks((0,50,100))
ax.set_xticklabels((0,50,100))
ax.set_yticks((0,5,10))
ax.set_yticklabels((0,5,10))
ax.set_xlabel("% of trials active")
ax.set_ylabel("mean activity\nwhen active (sd)")
layout.insert_figures('plots')
layout.write_svg(outputFolder / "fractionActiveTrials.svg")