import numpy as np
import pandas as pd
import seaborn as sns
import cmocean
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FixedLocator
from utils import readSessions, fancyViz, sessionBarPlot
from collections import defaultdict
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
layout = figurefirst.FigureLayout(templateFolder / "openFieldNew.svg")
layout.make_mplfigures()

longGtNames = {'d1':'D1-Cre','a2a':'A2A-Cre','oprm1':'Oprm1-Cre'}

#%%
tuningData = analysisOpenField.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005

#%%
ex_session = ('oprm1','5308','190224')
s = next(readSessions.findSessions(endoDataPath, genotype=ex_session[0],
                                   animal=ex_session[1], date=ex_session[2],
                                   task='openField'))
traces = s.readDeconvolvedTraces(zScore=True)
ex_tunings = tuningData.query("genotype=='{}' & animal=='{}' & date=='{}'".format(*ex_session))

example_sess = {'genotype':'oprm1', 'animal':'5308', 'date':'190224'}
#example_sess = {'genotype':'oprm1', 'animal':'5703', 'date':'190201'}
coords = analysisOpenField.getSmoothedOFTTracking(endoDataPath, example_sess['genotype'],
                                example_sess['animal'], example_sess['date'])


#%%

for i in np.arange(4):
    ax = layout.axes['ex_path_{}'.format(i+1)]['axis']
    start, end = 3200+i*400, 3200+(i+1)*400
    cdf = coords.loc[start:end].copy()
    ax.plot(cdf.x, cdf.y, color='k', lw=0.5, zorder=0)
    for (actioNo, behavior), adf in cdf.groupby(['actionNo', 'behavior']):
        ax.plot(adf.x, adf.y, c=style.getColor(behavior), lw=5, alpha=.43,
                 zorder=2)
    x = cdf.iloc[0].x
    y = cdf.iloc[0].y
    circ = mpatches.Circle((x, y), 4, color='k', zorder=3)
    ax.text(x,y,i+1,color="w", ha="center", va="center", fontsize=6)
    ax.add_patch(circ)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(0,49)
    ax.set_ylim(0,49)

best_neurons = ex_tunings.set_index("neuron").groupby("action").tuning.idxmax()
order = ["running", "leftTurn", "rightTurn"]
for p, behavior in enumerate(order):
    max_neuron = best_neurons[behavior]
    trace = traces[max_neuron]
    
    axfv = layout.axes['ex_{}'.format(p+1)]['axis']
    fv = fancyViz.OpenFieldSchematicPlot(s, linewidth=mpl.rcParams['axes.linewidth'])
    img = fv.draw(trace, ax=axfv)
    
    axbg = layout.axes['ex_{}_bg'.format(p+1)]['axis']
    axbg.axvspan(-.055, -.03, .1, .93, color=style.getColor(behavior), alpha=1,
                 clip_on=False)
    axbg.set_xlim((0,1))
    axbg.set_axis_off()
    
cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
#cax.xaxis.tick_top()
#cax.tick_params(axis='both', which='both',length=0)
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})

start = 3200
start = start / 20.0
tpr = 24
for r in np.arange(4):
    axt = layout.axes['ex_t{}'.format(r+1)]['axis']
    for behavior in order:
        max_neuron = best_neurons[behavior]
        axt.vlines(traces.loc[start:start+tpr, max_neuron].index,
                   0, traces.loc[start:start+tpr, max_neuron],
                   lw=.35, clip_on=False, color=style.getColor(behavior)) 
        
    for behavior in order:
        axt.fill_between(traces.index.values, 11, -1,              
                         where=coords['behavior'] == behavior,
                         color=style.getColor(behavior), lw=0, alpha=.15)
    axt.plot([start+.4],[9],'ko', ms=6)
    axt.text(start+.4,8.7,r+1,color="w", ha="center", va="center", fontsize=6)
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
patches = [mpatches.Patch(color=style.getColor(b), label=t, alpha=.15) 
               for b,t in [('running','running'),('leftTurn','left turn'),('rightTurn','right turn')]]
axt.legend(handles=patches, ncol=3, mode='expand', bbox_to_anchor=(0,1.02,1,1.02),
           loc='lower center')

ax = layout.axes['tuning_fov']['axis']

df = ex_tunings.loc[ex_tunings.groupby('neuron').tuning.idxmax()].copy()
df['color'] = df.action
df.loc[~df.signp, 'color'] = 'none'
df['color'] = df.color.apply(lambda c: np.array(style.getColor(c)))

rois = s.readROIs()
rois = np.array([rois[n].unstack('x').values for n in rois])
#sel_cnts = analysisTunings.get_centers(rois)[sel_neurons]

rs = []
for roi, color in zip(rois, df.color.values):
    roi /= roi.max()
    roi = roi**1.5
    roi = np.clip(roi-.1, 0, .85)
    roi /= roi.max()
    r = np.array([(roi > 0).astype('int')]*3) * color[:, np.newaxis, np.newaxis]
    r = np.concatenate([r, roi[np.newaxis]], axis=0)
    rs.append(r.transpose((1,2,0)))    
rs = np.array(rs)

for img in rs:
    ax.imshow(img)
#ax.scatter(sel_cnts[:,0], sel_cnts[:,1], marker='o', edgecolor='k', facecolor='none', 
#           s=25, alpha=1, lw=mpl.rcParams['axes.linewidth'])

ax.axis('off')


#%%
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


#%% pie charts
df = tuningData.copy()

# only keep max tuning for each neuron
maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
maxdf.loc[~df.signp, 'action'] = 'blank' # don't color if not significant
maxdf = maxdf.groupby(['genotype','action'])[['signp']].count() # get counts

for g in ['d1','a2a','oprm1']:
    ax = layout.axes['pie_{}'.format(g)]['axis']
    gdata = maxdf.loc[g]   
    ws, ts = ax.pie(gdata.values.squeeze(), wedgeprops={'lw':0, 'edgecolor':'w'},
                    explode=[.1]*len(gdata),
                    textprops={'color':'k'}, colors=[style.getColor(b) for b in gdata.index])

    ax.set_aspect('equal')


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
    deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
    genotypes.append(np.full(deconv.shape[1], sess.meta.genotype))
    sessTimes = times.loc[str(sess)]
    onsets.append(deconv.groupby(sessTimes.onsetTimes).mean().reindex(np.arange(-20,20)).T)
    offsets.append(deconv.groupby(sessTimes.offsetTimes).mean().reindex(np.arange(-20,20)).T)
genotypes = np.concatenate(genotypes)
onsets = pd.concat(onsets)
offsets = pd.concat(offsets)

for fullDf, axName in [(onsets, "avgOnset"), (offsets, "avgOffset")]:
    ax = layout.axes[axName]["axis"]
    for gt, df in fullDf.groupby(genotypes):
        m = df.mean(axis=0)
        err = df.sem(axis=0)
        ax.fill_between(m.index*0.05, m-err, m+err, color=style.getColor(gt), alpha=0.25)
        ax.plot(m.index*0.05, m, color=style.getColor(gt))
    ax.axvline(0, ls='--', c='k')
    ax.set_ylim(-0.05, 0.08)
    sns.despine(ax=ax)
layout.axes["avgOffset"]["axis"].set_yticks([])
layout.axes["avgOnset"]["axis"].set_title("movement\nonset (s)")
layout.axes["avgOffset"]["axis"].set_title("movement\noffset (s)")
layout.axes["avgOnset"]["axis"].set_ylabel("z-score")


moveIncr = analysisOpenField.populationIncreaseInMovement(endoDataPath)
moveIncr["diff"] = moveIncr.meanMoving - moveIncr.meanStationary
ax = layout.axes["corrWithSpeed"]["axis"]
order = ["d1", "d1_shuffled", "a2a", "a2a_shuffled", "oprm1", "oprm1_shuffled"]
def colorFunc(l):
    if l.endswith("_shuffled"): return 'k'
    else: return style.getColor(l)
sessionBarPlot.sessionBarPlot(moveIncr, "diff", ax, colorFunc, genotypeOrder=order,
                              weightScale=0.05)
ax.set_xticks([0.5, 2.5, 4.5])
ax.set_xticklabels(["D1", "A2a", "Oprm1"])
#ax.set_ylabel("$\langle S \rangle_{mov} - \langle S \rangle_{still}$")
ax.set_ylabel("increase")
ax.set_ylim(-0.03, 0.1)
sns.despine(ax=ax)

#%%
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

for g in ['d1','a2a','oprm1']:
    ax.errorbar(avg_s[g], avg[g], xerr=sem_s[g], yerr=sem[g],
                color=style.getColor(g), fmt='s', markersize=3,
                markeredgewidth=mpl.rcParams['axes.linewidth'],
                markeredgecolor='k', ecolor='k',
                label={'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])

ax.plot([15,65],[15,65], ls=':', color='k', alpha=.5, zorder=-1)    

ax.set_xlim((15,65))
ax.set_ylim((15,65))
ax.set_xticks(np.arange(15,66,25))
ax.set_yticks(np.arange(15,66,25))
ax.set_aspect('equal')
ax.set_xlabel('expected')
ax.set_ylabel('observed')
ax.text(40, 65, 'μm to nearest\ntuned neighbor', ha='center', va='center',
        fontdict={'fontsize':7})
ax.legend(loc='lower right', bbox_to_anchor=(1.1, .05))
ax.set_aspect('equal')
sns.despine(ax=ax)

#%% Decoding
genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running',
                 'leftTurn': 'left turn', 'rightTurn': 'right turn'}
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
decodingData = analysisOpenField.decodeWithIncreasingNumberOfNeurons(endoDataPath, segmentedBehavior)
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
plt.legend(meanHandles+[shuffleHandle], [genotypeNames[g] for g in order]+["shuffled",],
           loc=(0.45, 0.45), ncol=2)

plt.ylim(0,1)
plt.xlim(0,300)
plt.xlabel("number of neurons")
plt.ylabel("decoding accuracy (%)")
plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
sns.despine(ax=plt.gca())

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "openFieldNew.svg")
print("Done")

