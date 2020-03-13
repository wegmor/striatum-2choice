import numpy as np
import pandas as pd
import seaborn as sns
import cmocean
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from utils import readSessions, fancyViz, roiPlot
import pathlib
import figurefirst
import style
import analysisOpenField
plt.ioff()

#%%
style.set_context()
endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "figure1OpenField.svg")
layout.make_mplfigures()

longGtNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running',
                 'leftTurn': 'left turn', 'rightTurn': 'right turn'}
behaviorOrder = ("leftTurn", "rightTurn", "running", "stationary")
genotypeOrder = ("d1", "a2a", "oprm1")

#%%
tuningData = analysisOpenField.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005

#%%
exampleSessParams = {'genotype':'oprm1', 'animal':'5308', 'date':'190224'}
exampleSess = next(readSessions.findSessions(endoDataPath, task='openField', **exampleSessParams))
traces = exampleSess.readDeconvolvedTraces(rScore=True)
queryStr = " & ".join(["{}=='{}'".format(*v) for v in exampleSessParams.items()])
ex_tunings = tuningData.query(queryStr)

#%% Example of deconvolution
traceExampleNeurons = [194, 166, 142, 13, 6]

caTraces = exampleSess.readCaTraces()[traceExampleNeurons]
caTraces -= caTraces.min(axis=0)
caTraces /= caTraces.std(axis=0)

ax = layout.axes['ex_fov']['axis']
rois = exampleSess.readROIs()
colors = np.full((rois.shape[1], 3), 0.75)
for i, n in enumerate(traceExampleNeurons):
    colors[n, :] = mpl.colors.to_rgb("C{}".format(i))    
roiPlot.roiPlot(rois, colors, ax)
ymax, xmax = rois[0].unstack().shape
ax.plot([40, 120], [ymax, ymax], 'k', lw=mpl.rcParams['axes.linewidth'])
ax.text(80, ymax+3, "200μm", ha="center", va="top", fontsize=6)
ax.set_xlim((0, xmax))
ax.set_ylim((ymax, 0))
ax.axis("equal")
ax.axis('off')

ax = layout.axes['ex_deconv']['axis']
timeslice = slice(1050, 1250)
for i, n in enumerate(traceExampleNeurons):
    color = 'C{}'.format(i)
    bottom = traces[n].iloc[timeslice].min()
    ax.vlines(traces.index[timeslice], 12*i+bottom, 12*i+traces[n].iloc[timeslice],
              color=color, lw=.5)
    ax.plot(caTraces[n].iloc[timeslice] + 12*i, color=color, lw=1, alpha=0.5)
    
starttime = timeslice.start / 20.0
ax.plot(starttime + np.array([1.5, -0.5, -0.5]), [60, 60, 60-6],
        'k', lw=mpl.rcParams['axes.linewidth'])
ax.text(starttime-0.65, 60-3, '6sd', ha='right', va='center', fontsize=6)
ax.text(starttime+0.5, 60.5, '2s', ha='center', va='bottom', fontsize=6)
ax.axis("off")

#%% Population mean on movement onset and offset
def getOnAndOffset(s):
    s = s[["startFrame", "stopFrame", "behavior"]]
    s["nextBehavior"] = s.behavior.shift(-1)
    s["prevBehavior"] = s.behavior.shift(1)
    start = s.startFrame.iloc[0]
    end = s.stopFrame.iloc[-1]
    s = s.set_index("startFrame", drop=False).rename_axis(index="frame")
    s = s.reindex(np.arange(start, end), method="ffill")
    s["framesSinceStart"] = s.index - s.startFrame
    s["framesTillEnd"] = s.stopFrame - s.index

    onsetTimes = np.full(len(s), np.nan)
    m = s.behavior=='stationary'
    onsetTimes[m] = -s.framesTillEnd[m]
    m = s.prevBehavior=='stationary'
    onsetTimes[m] = s.framesSinceStart[m]

    offsetTimes = np.full(len(s), np.nan)
    m = s.nextBehavior=='stationary'
    offsetTimes[m] = -s.framesTillEnd[m]
    m = s.behavior=='stationary'
    offsetTimes[m] = s.framesSinceStart[m]
    return pd.DataFrame({'onsetTimes': onsetTimes, 'offsetTimes': offsetTimes}, s.index)

segmented = analysisOpenField.segmentAllOpenField(endoDataPath)
times = segmented.groupby(level=[0,1]).apply(getOnAndOffset).droplevel(1)
onsets = []
offsets = []
genotypes = []
for sess in readSessions.findSessions(endoDataPath, task="openField"):
    deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    genotypes.append(np.full(deconv.shape[1], sess.meta.genotype))
    sessTimes = times.loc[str(sess)]
    onsets.append(deconv.groupby(sessTimes.onsetTimes).mean().reindex(np.arange(-20,20)).T)
    offsets.append(deconv.groupby(sessTimes.offsetTimes).mean().reindex(np.arange(-20,20)).T)
genotypes = np.concatenate(genotypes)
onsets = pd.concat(onsets)
offsets = pd.concat(offsets)
for fullDf, axName in [(onsets, "avgOnset"), (offsets, "avgOffset")]:
    ax = layout.axes[axName]["axis"]
    for gt in genotypeOrder:
        df = fullDf[genotypes==gt]
        m = df.mean(axis=0)
        err = df.sem(axis=0)
        ax.fill_between(m.index*0.05, m-err, m+err, color=style.getColor(gt), alpha=0.25, lw=0)
        ax.plot(m.index*0.05, m, color=style.getColor(gt), label=longGtNames[gt])
    ax.axhline(0, ls=':', c='k', lw=0.5, alpha=0.5)
    ax.set_ylim(-0.04, 0.08)
    ax.set_xlim(-1, 1)
    ax.set_yticks((-0.04, 0.08))
    ax.set_yticks((0, 0.04), minor=True)
    ax.set_xticks((-0.5, 0.5), minor=True)
    ax.set_ylabel("z-score", labelpad=-7)
    ax.set_xlabel("time (s)")
    sns.despine(ax=ax)
ax = layout.axes["avgOnset"]["axis"]
ax.axvspan(0, 1, color="gray", alpha=.2, zorder=-10, label="movement")
ax.legend(bbox_to_anchor=(-0.1, 1.1, 1.0, 0.1), columnspacing=1.2, ncol=2)
ax = layout.axes["avgOffset"]["axis"]
ax.axvspan(-1, 0, color="gray", alpha=.2, zorder=-10)

#%% Example path
coords = analysisOpenField.getSmoothedOFTTracking(endoDataPath, **exampleSessParams)
ax = layout.axes['ex_path']['axis']
start, end = 3200, 3200+2400
cdf = coords.loc[start:end].copy()
ax.plot(cdf.x, cdf.y, color='k', lw=0.5, zorder=0)
for (actioNo, behavior), adf in cdf.groupby(['actionNo', 'behavior']):
    ax.plot(adf.x, adf.y, c=style.getColor(behavior), lw=5, alpha=.43,
                 zorder=2)
ax.set_xticks(())
ax.set_yticks(())
ax.set_xlim(0,49)
ax.set_ylim(0,49)

#%% Example traces and fancyViz
best_neurons = ex_tunings.set_index("neuron").groupby("action").tuning.idxmax()
order = ["leftTurn", "rightTurn", "running"]
for p, behavior in enumerate(order):
    max_neuron = best_neurons[behavior]
    trace = traces[max_neuron]
    
    axfv = layout.axes['ex_{}'.format(p+1)]['axis']
    fv = fancyViz.OpenFieldSchematicPlot(exampleSess, linewidth=mpl.rcParams['axes.linewidth'])
    img = fv.draw(trace, ax=axfv)
    
    axbg = layout.axes['ex_{}_bg'.format(p+1)]['axis']
    axbg.axvspan(-.055, -.03, .1, .93, color=style.getColor(behavior), alpha=1,
                 clip_on=False)
    axbg.set_xlim((0,1))
    axbg.set_axis_off()
    
cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})
start = 3200
start = start / 20.0
tpr = 24
for r in np.arange(5):
    axt = layout.axes['ex_t{}'.format(r+1)]['axis']
    for behavior in order:
        max_neuron = best_neurons[behavior]
        axt.vlines(traces.loc[start:start+tpr, max_neuron].index,
                   0, traces.loc[start:start+tpr, max_neuron],
                   lw=.35, clip_on=False, color=style.getColor(behavior)) 
        
    for behavior in order:#+["stationary"]:
        axt.fill_between(traces.index.values, 11, -1,              
                         where=coords['behavior'] == behavior,
                         color=style.getColor(behavior), lw=0, alpha=.15)
    axt.set_ylim((-1,12))
    axt.set_xlim((start, start+tpr))
    axt.axis('off')
    start += tpr
y0=-2
sd=6
x1=start+.3
sec=2
axt.vlines(x1, y0, y0+sd, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
axt.text(x1+.25, y0+sd/2, '{}sd'.format(sd), ha='left', va='center',
         fontdict={'fontsize':6})
axt.hlines(y0, x1-sec, x1, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
axt.text(x1-sec/2, y0-1, '{}s'.format(sec), ha='center', va='top',
         fontdict={'fontsize':6})

axt = layout.axes['ex_t1']['axis']
patches = [mpatches.Patch(color=style.getColor(b), label=behaviorNames[b],
                          alpha=.15) for b in order]
axt.legend(handles=patches, ncol=3, mode='expand', bbox_to_anchor=(0,1.02,1,1.02),
           loc='lower center')

#%% Tuning example histogram
ax = layout.axes['tuning_hist1']['axis']
hdata = tuningData.query('genotype == "oprm1" & action == "leftTurn"').copy()

ax.hist(hdata['tuning'], bins=np.arange(-20,40,1), lw=0, color='gray', alpha=.6,
        histtype='stepfilled')
ax.hist(hdata.loc[hdata.signp,'tuning'], np.arange(-20,40,1), lw=0,
        histtype='stepfilled', color=style.getColor('leftTurn'))

ax.text(21,45,'significant\ntuning',ha='right',va='bottom',fontdict={'fontsize':7},
        color=style.getColor('leftTurn'))
ax.text(4.5,200,'left turn tuning',ha='center',va='center',fontdict={'fontsize':7})
ax.set_yticks((0,100,200))
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.set_xticks((-5,0,5,10,15))
ax.set_xlim((-8,17))
ax.set_ylim((0,200))
ax.set_xlabel('tuning score')
ax.set_ylabel('# neurons')
sns.despine(ax=ax)

#%% Pie charts
df = tuningData.copy()

# only keep max tuning for each neuron
maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~df.signp, 'action'] = 'blank' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

for g in genotypeOrder:
    ax = layout.axes['pie_{}'.format(g)]['axis']
    gdata = maxdf.loc[g].loc[['leftTurn', 'rightTurn', 'blank', 'running', 'stationary']]
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[style.getColor(b) for b in gdata.index])

    ax.set_aspect('equal')

#%% Tuned PSTHs
ewindows = analysisOpenField.getEventWindows(endoDataPath, behaviorOrder)
ewindows.set_index(["animal", "date", "neuron", "label"], inplace=True)
ewindows["tuned"] = tuningData.set_index(["animal", "date", "neuron", "action"]).pct>0.0995
ewindows.reset_index(inplace=True)

for label in ("leftTurn", "running"):#behaviorOrder:
    ax = layout.axes["onset_"+label]["axis"]
    for gt in genotypeOrder:
        mask = np.logical_and(ewindows.genotype==gt, ewindows.label==label)
        mask = np.logical_and(mask, ewindows.tuned)
        ma = ewindows[mask].groupby(["animal", "date", "neuron"]).mean()["frameNo"]
        m = ma.mean()
        s = ma.sem()
        ax.plot(np.linspace(-1,1,40), m, color=style.getColor(gt))
        ax.fill_between(np.linspace(-1,1,40), m-s, m+s, color=style.getColor(gt), alpha=0.15, lw=0)
    #ax.axvline(0, color='k', linestyle='--', lw=0.75)
    ax.axvspan(0, 1, color=style.getColor(label), alpha=.15, zorder=-10)
    ax.set_title(behaviorNames[label] + ' tuned', pad=4)
    ax.set_xlim(-1, 1)
    ax.set_xticks((-1,0,1))    
    ax.set_xticks((-0.5, 0.5), minor=True)
    if label == "running" or label == "stationary":
        ax.set_xlabel("time (s)")#("time from\n"+longerName[label]+" onset (s)")
    else:
        ax.set_xticklabels(["","",""])
    if label == "rightTurn" or label == "stationary":
        ax.set_yticks([])
        ax.set_yticks(np.arange(-0.05, 0.25, 0.05), minor=True)
    else:
        ax.set_yticks((-0.05, 0.20))
        ax.set_yticks(np.arange(0, 0.2, 0.05), minor=True)
        ax.set_ylabel("z-score", labelpad=-7)
    ax.set_ylim(-0.05, 0.20)
    ax.axhline(0, ls=':', c='k', lw=0.5, alpha=0.5)
    sns.despine(ax=ax)
lines = [mpl.lines.Line2D([], [], color=style.getColor(gt)) for gt in genotypeOrder]
labels = [longGtNames[gt] for gt in genotypeOrder]
layout.axes["onset_running"]["axis"].legend(lines, labels, ncol=3, columnspacing=1.0,
                                            bbox_to_anchor=(0.25, 1.4, 1, 0.1))

#%% Example tuning FOV
ax = layout.axes['tuning_fov']['axis']
df = ex_tunings.loc[ex_tunings.groupby('neuron').tuning.idxmax()]
colors = df.action.copy()
colors[~df.signp] = 'none'
colors = np.array([style.getColor(c) for c in colors])
rois = exampleSess.readROIs()
roiPlot.roiPlot(rois, colors, ax)
sel_cnts = np.array(list(rois[best_neurons[order]].idxmax(axis=0)))
ax.scatter(sel_cnts[:,1], sel_cnts[:,0], marker='o', edgecolor='k', facecolor='none', 
           s=8, alpha=1, lw=mpl.rcParams['axes.linewidth'])
ax.axis('off')

#%% Distance between tuned neurons
pdists = analysisOpenField.getPDistData(endoDataPath, tuningData)
ax = layout.axes['dist_scatter']['axis']

for g, gdata in pdists.groupby('genotype'):
    ax.scatter(gdata.dist_shuffle, gdata.dist, s=gdata.noNeurons/25,
               edgecolor=style.getColor(g), facecolor=style.getColor(g),
               alpha=.4, lw=mpl.rcParams['axes.linewidth'])
    
avg = pdists.groupby('genotype').apply(analysisOpenField.wAvg, 'dist', 'noNeurons')
avg_s = pdists.groupby('genotype').apply(analysisOpenField.wAvg, 'dist_shuffle', 'noNeurons')
sem = pdists.groupby('genotype').apply(analysisOpenField.bootstrap, 'dist', 'noNeurons')
sem_s = pdists.groupby('genotype').apply(analysisOpenField.bootstrap, 'dist_shuffle', 'noNeurons')

for g in genotypeOrder:
    ax.errorbar(avg_s[g], avg[g], xerr=sem_s[g], yerr=sem[g],
                color=style.getColor(g), fmt='s', markersize=3,
                markeredgewidth=mpl.rcParams['axes.linewidth'],
                markeredgecolor='k', ecolor='k',
                label=longGtNames[g])

ax.plot([15,65],[15,65], ls=':', color='k', alpha=.5, zorder=-1)    

ax.set_xlim((15,65))
ax.set_ylim((15,65))
ax.set_xticks(np.arange(15,66,25))
ax.set_yticks(np.arange(15,66,25))
ax.set_xticks(np.arange(15,66,5), minor=True)
ax.set_yticks(np.arange(15,66,5), minor=True)
ax.set_aspect('equal')
ax.set_xlabel('expected')
ax.set_ylabel('observed')
ax.text(40, 65, 'μm to nearest\ntuned neighbor', ha='center', va='center',
        fontdict={'fontsize':7})
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-3:], labels[-3:], loc='lower right', bbox_to_anchor=(1.1, .05))
ax.set_aspect('equal')
sns.despine(ax=ax)


#%% Decoding
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
decodingData = analysisOpenField.decodeWithIncreasingNumberOfNeurons(endoDataPath)
decodingData.insert(1, "genotype", decodingData.session.str.split("_").str[0])

ax = layout.axes["decodeWithIncreasingNumberOfNeurons"]["axis"]
for strSess, df in decodingData.groupby("session"):
    genotype = strSess.split("_")[0]
    ax.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=0.2, lw=.5)
    ax.plot(df.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
             alpha=0.2, lw=.5)
for genotype, df in decodingData.groupby("genotype"):
    ax.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=1.0)
ax.plot(decodingData.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
         alpha=1.0)

meanHandles = [mpl.lines.Line2D([], [], color=style.getColor(g)) for g in genotypeOrder]
shuffleHandle = mpl.lines.Line2D([], [], color=style.getColor("shuffled"))
ax.legend(meanHandles+[shuffleHandle],
          [longGtNames[g] for g in genotypeOrder]+["shuffled"],
          loc=(0.2, 0.9), ncol=2, columnspacing=1.2)
ax.set_xlim(0,300)
ax.set_ylim(0,1)
ax.set_xlabel("number of neurons")
ax.set_ylabel("accuracy (%)")
ax.set_xticks(np.arange(50, 300, 50), minor=True)
ax.set_yticks(np.linspace(0,1,5))
ax.set_yticklabels(np.linspace(0,100,5,dtype=np.int64))
ax.set_title("SVM decoding of behavior", pad=12)
sns.despine(ax=ax)

#%% Confusion matrices
decodingData = analysisOpenField.decodingConfusion(endoDataPath)
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]
colorMap = mpl.colors.ListedColormap([style.getColor(b) for b in behaviorOrder])
for gt, data in decodingData.groupby("genotype"):
    weightedData = data.set_index(["true", "predicted"]).eval("occurencies * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    ax = layout.axes["confusionMatrix_{}".format(gt)]["axis"]
    img = ax.pcolormesh(weightedData[list(behaviorOrder)].reindex(list(behaviorOrder)),
                        edgecolors='w', vmin=0, vmax=1, cmap=cmocean.cm.amp, linewidth=.35)
    ax.set_xlabel("predicted" if gt=="a2a" else None, labelpad=2)
    ax.set_ylabel("truth" if gt=="d1" else None)
    ax.set_title(longGtNames[gt], pad=1)
    ax.set_ylim(4.5, 0)
    ax.set_xlim(-0.5, 4)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])
    X, Y = np.meshgrid([-0.5, 0], np.arange(5))
    ax.pcolormesh(X, Y, np.arange(4).reshape(4,1), linewidth=.35, edgecolors='w', cmap=colorMap)
    X, Y = np.meshgrid(np.arange(5), [4, 4.5])
    ax.pcolormesh(X, Y, np.arange(4).reshape(1,4), linewidth=.35, edgecolors='w', cmap=colorMap)
cax = layout.axes["colorbar_confusionMatrix"]["axis"]
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.axis("off")
cax.text(-0.025, .4, '0%', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.025, .4, '100%', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0.5, 1.1, 'recall', ha='center', va='bottom', fontdict={'fontsize':6})

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "figure1OpenField.svg")
