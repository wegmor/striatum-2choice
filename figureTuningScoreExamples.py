import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FixedLocator
from utils import readSessions, fancyViz
from collections import defaultdict
import pathlib
import figurefirst
import style
import analysisTunings
plt.ioff()

style.set_context()
endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")
if not outputFolder.is_dir():
    outputFolder.mkdir()
layout = figurefirst.FigureLayout(templateFolder / "tuningScoreExamples.svg")
layout.make_mplfigures()


tuningData = analysisTunings.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005

histAx = layout.axes['tuningHistogram']['axis']
hdata = tuningData.query('genotype == "oprm1" & action == "mC2L-"').copy()

cmap = plt.cm.get_cmap("RdYlBu")

histAx.hist(hdata['tuning'], bins=np.arange(-20,40,1), lw=1, color=cmap(0.5), alpha=1.0,
        histtype='stepfilled')
histAx.hist(hdata.loc[hdata.signp,'tuning'], np.arange(-20,40,1), lw=0,
        histtype='stepfilled', color=cmap(0.0))
histAx.hist(hdata.loc[hdata.signn,'tuning'], np.arange(-20,40,1), lw=0,
        histtype='stepfilled', color=cmap(1.0))
histAx.text(11,85,'significant\npositive tuning',ha='right',va='bottom',fontdict={'fontsize':7},
        color=cmap(0.0))
histAx.text(-7,85,'significant\nnegative tuning',ha='right',va='bottom',fontdict={'fontsize':7},
        color=cmap(1.0))
histAx.axvline(0,color="black", linestyle='--')
histAx.set_yticks((0,200,400))
histAx.yaxis.set_minor_locator(MultipleLocator(100))
histAx.set_xticks([])
#ax.set_xticks((-15,0,15,30))
histAx.set_xlim((-15,40))
histAx.set_ylim((0,400))
#ax.set_xlabel('tuning score')
histAx.set_ylabel('# neurons')
sns.despine(ax=histAx)

saturation = 1
#sampleBins = pd.cut(hdata.tuning, np.arange(-15,40,4))
#samples = hdata.groupby(sampleBins).apply(lambda df: df.iloc[len(df)//2])
#ax.set_xticks(samples.tuning)
#ax.set_xticklabels([])
sortedHdata = hdata.set_index("tuning").sort_index()
positions = np.linspace(-11.5, 36, 13)
actualPositions = []
for i in range(13):
    ax = layout.axes['histEx_{}'.format(i+1)]['axis']
    sample = sortedHdata.loc[positions[i]:].iloc[0]
    actualPositions.append(sample.name)
    sess = next(readSessions.findSessions(endoDataPath, animal=sample.animal,
                                          date=sample.date, task="2choice"))
    fv = fancyViz.SchematicIntensityPlot(sess, splitReturns=False,
                                         linewidth=mpl.rcParams['axes.linewidth'],
                                         smoothing=7, saturation=saturation)
    trace = sess.readDeconvolvedTraces()[sample.neuron]
    trace -= trace.mean()
    trace /= trace.std()
    fv.draw(trace, ax=ax)
histAx.set_xticks(actualPositions)
ax.set_xticklabels([])
cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(cax=cax, orientation='vertical')
#cax.xaxis.tick_top()
#cax.tick_params(axis='both', which='both',length=0)
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(0.5, -0.02, str(-saturation), ha='center', va='top',
         fontdict={'fontsize':6}, transform=cax.transAxes)
cax.text(0.5, 1.00, str(saturation), ha='center', va='bottom',
         fontdict={'fontsize':6}, transform=cax.transAxes)
cax.text(-0.5, 0.5, 'z-score', ha='right', va='center',
         fontdict={'fontsize':6}, rotation=90, transform=cax.transAxes)

#%% Figure B
for gt in ("d1", "a2a", "oprm1"):
    fvs = {}
    traces = {}
    for sess in readSessions.findSessions(endoDataPath, genotype=gt, task="2choice"):
        fvs[str(sess)] = fancyViz.SchematicIntensityPlot(sess, splitReturns=False,
                                             linewidth=mpl.rcParams['axes.linewidth'],
                                             smoothing=7, saturation=saturation)
        traces[str(sess)] = sess.readDeconvolvedTraces(zScore=True)

    nTunings = tuningData[tuningData.genotype==gt].set_index(["genotype", "animal", "date", "neuron", "action"]).signp
    nTunings = nTunings.groupby(level=[0,1,2,3]).sum().sort_values().rename("nTunings").reset_index()
    nExamples = [0,8,8,4,1,1,1,1]
    for nt, ne in enumerate(nExamples):
        sample = nTunings[nTunings.nTunings == nt].sample(ne)
        for i in range(ne):
            r = sample.iloc[i]
            ax = layout.axes['nTuningsEx_{}_{}_{}'.format(r.genotype, nt, i+1)]['axis']
            sess = "{}_{}_{}".format(r.genotype, r.animal, r.date)
            fvs[sess].draw(traces[sess][r.neuron], ax=ax)

layout.insert_figures('plots')
layout.write_svg(outputFolder / "tuningScoreExamples.svg")