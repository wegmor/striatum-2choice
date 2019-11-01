import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import analysisTunings
import figurefirst
import itertools
import tqdm
from collections import defaultdict
from utils import readSessions, fancyViz
from sklearn.metrics import silhouette_samples
import style
plt.ioff()

#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
    
#%%
layout = figurefirst.FigureLayout(templateFolder / "behaviorSupp.svg")
layout.make_mplfigures()

#%%
dfs = []
for sess in tqdm.tqdm(readSessions.findSessions(endoDataPath, task="2choice"),
                      total=66):
    lfa = sess.labelFrameActions(reward="fullTrial", switch=True)
    df = lfa.groupby(["label", "actionNo"]).actionDuration.mean().reset_index()
    dfs.append(df.assign(genotype=sess.meta.genotype,
                         animal=sess.meta.animal,
                         date=sess.meta.date))
allDurations = pd.concat(dfs)
allDurations["mainLabel"] = allDurations.label.str[:4]
allDurations["trialType"] = allDurations.label.str[4:]
allDurations.actionDuration /= 20.0

allDurations = allDurations[np.logical_or(allDurations.label.str[-2] != 'd', 
                                          allDurations.actionDuration != 0.35)]
allDurations.loc[allDurations.label.str.match("p[LR]2C[or][\.!]"), "actionDuration"] += 0.35
allDurations.label = allDurations.label.str.replace("d", "o")
allDurations.trialType = allDurations.trialType.str.replace("d", "o")

labels = ["mC2L", "mC2R", "mL2C", "mR2C", "pC2L", "pC2R", "pL2C", "pR2C"]
palette = {gt: style.getColor(gt) for gt in ("d1", "a2a", "oprm1")}
titles = {"mC2L": "center-to-left", "mC2R": "center-to-right",
          "mL2C": "left-to-center", "mR2C": "right-to-center",
          "pC2L": "center port (going left)", "pC2R": "center port (going right)",
          "pL2C": "left port", "pR2C": "right port"}
xlabels = ["win\nstay", "win\nswitch", "lose\nstay", "lose\nswitch"]
for label in labels:
    ax = layout.axes['duration_'+label]['axis']
    sns.boxplot(data=allDurations[allDurations.mainLabel==label],
                x="trialType", y="actionDuration", hue="genotype",
                ax=ax, order=("r.", "r!", "o.", "o!"), fliersize=0,
                hue_order=("d1", "a2a", "oprm1"), palette=palette)
    ax.set_title(titles[label], pad=6)
    if label[0]=='p' and label[1] in "LR":
        ax.set_ylim(0, 5)
    else:
        ax.set_ylim(0, 1.5)
    sns.despine(ax=ax)
    
    if label=="mC2L":
        handles, legendLabels = ax.get_legend_handles_labels()
    if label in ("mC2L", "pC2L", "pL2C"):
        ax.set_ylabel("duration (s)")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.legend([])
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("")
    if label.startswith("p"):
        ax.axhline(0.35, ls="--", color="k")
ax = layout.axes['duration_legend']['axis']
ax.legend(handles, legendLabels, title="genotype")
ax.axis("off")
layout.insert_figures('plots')
layout.write_svg(outputFolder / "behaviorSupp.svg")