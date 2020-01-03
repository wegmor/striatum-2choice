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
ax = layout.axes['ex_path']['axis']
coords['distance'] = np.sqrt(coords.x.diff()**2 + coords.y.diff()**2)
start, end = 3200, 5600
cdf = coords.loc[start:end].copy()
cdf['distance'] = cdf.distance.cumsum().fillna(0)

ax.plot(cdf.x, cdf.y, color='k', lw=0.5, zorder=0)
timepoints = np.linspace(start, end, 5, endpoint=False, dtype=np.int)
for i in np.arange(5):
    x = coords.iloc[timepoints[i]].x
    y = coords.iloc[timepoints[i]].y
    circ = mpatches.Circle((x, y), 2, color='k', zorder=3)
    ax.text(x,y,i+1,color="w", ha="center", va="center", fontsize=6)
    ax.add_patch(circ)
'''
for c in cdf.groupby(pd.cut(cdf.distance, 5)).last().dropna().itertuples():
    x = c.x + 1.5 * np.cos(np.deg2rad(c.bodyDirection))
    y = c.y + 1.5 * np.sin(np.deg2rad(c.bodyDirection))
    w = mpatches.Wedge((x, y), 4, c.bodyDirection-180-22, c.bodyDirection-180+22,
              facecolor='w', edgecolor='none', lw=2, alpha=1, zorder=1)
    plt.gca().add_patch(w)
    w = mpatches.Wedge((x, y), 4, c.bodyDirection-180-22, c.bodyDirection-180+22,
              facecolor='none', edgecolor='k', lw=2, alpha=1, zorder=3)
    plt.gca().add_patch(w)
''' 
    
for (actioNo, behavior), adf in cdf.groupby(['actionNo', 'behavior']):
    ax.plot(adf.x, adf.y, c=style.getColor(behavior), lw=5, alpha=.43,
             zorder=2)

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

start = start / 20.0
tpr = 24
for r in np.arange(5):
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

fvs = {gt: fancyViz.OpenFieldSchematicPlot(s, linewidth=mpl.rcParams['axes.linewidth']) for gt in ["d1", "a2a", "oprm1"]}
for sess in readSessions.findSessions(endoDataPath, task='openField'):
    traces = sess.readDeconvolvedTraces(zScore=True)
    fvs[sess.meta.genotype].setSession(sess)
    for n in traces.columns:
        fvs[sess.meta.genotype].addTraceToBuffer(traces[n])
for gt in fvs.keys():
    ax = layout.axes['avg_{}'.format(gt)]['axis']
    fvs[gt].drawBuffer(ax=ax)
cax = layout.axes['colorbar_avg']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
#cax.xaxis.tick_top()
#cax.tick_params(axis='both', which='both',length=0)
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})
#%% map

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
    
#%%
gby = tuningData.groupby(["genotype", "animal", "date", "action"])
df = pd.DataFrame({'nNeurons': gby.size(),
                   'percTuned': gby.signp.mean()*100}).reset_index()
order = ("leftTurn", "rightTurn", "running", "stationary")
for gt, subset in df.groupby("genotype"):
    ax = layout.axes['tunedFrac_'+gt]['axis']
    sessionBarPlot.actionBarPlot(subset, "percTuned", "action", ax, style.getColor, 
                                  weightCol="nNeurons", hueOrder=order, barAlpha=0.5)
    ax.set_ylim(0,75)
    if gt == "d1":
        ax.set_ylabel("tuned neurons (%)")
        ax.set_yticks([0,25,50,75])
    else:
        ax.set_yticks([])
    ax.set_title(longGtNames[gt])
    sns.despine(ax=ax)
'''

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


#%% TSNE
tuningTsne = analysisTunings.getTSNEProjection(tuningData)

#%%
for g,gdata in tuningTsne.groupby('genotype'):
    ax = layout.axes['tsne_'+g]['axis']
    
    ax.scatter(gdata[0], gdata[1],
               c=gdata.action.str.slice(0,4).apply(style.getColor),
               marker='.', alpha=.75, s=1.35, lw=0, clip_on=False)

    ax.set_xlim((tuningTsne[0].min(), tuningTsne[0].max()))
    ax.set_ylim((tuningTsne[1].min(), tuningTsne[1].max()))
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.axis('off')

ax = layout.axes['tsne_tuning']['axis']

ax.scatter(tuningTsne[0], tuningTsne[1],
           c=tuningTsne.action.str.slice(0,4).apply(style.getColor),
           marker='.', alpha=.75, s=3, lw=0, clip_on=False)

ax.set_xlim((tuningTsne[0].min(), tuningTsne[0].max()))
ax.set_ylim((tuningTsne[1].min(), tuningTsne[1].max()))
ax.invert_xaxis()
ax.set_aspect('equal')
ax.axis('off')


#%% similar tuning == closer spatially?
pdists = analysisTunings.getPDistData(endoDataPath, tuningData)
    
#%%
ax = layout.axes['dist_scatter']['axis']

for g, gdata in pdists.groupby('genotype'):
    ax.scatter(gdata.dist_shuffle, gdata.dist, s=gdata.noNeurons/25,
               edgecolor=style.getColor(g), facecolor=style.getColor(g),
               alpha=.4, lw=mpl.rcParams['axes.linewidth'])
    
avg = pdists.groupby('genotype').apply(analysisTunings.wAvg, 'dist', 'noNeurons')
avg_s = pdists.groupby('genotype').apply(analysisTunings.wAvg, 'dist_shuffle', 'noNeurons')
sem = pdists.groupby('genotype').apply(analysisTunings.bootstrap, 'dist', 'noNeurons')
sem_s = pdists.groupby('genotype').apply(analysisTunings.bootstrap, 'dist_shuffle', 'noNeurons')

for g in ['d1','a2a','oprm1']:
    ax.errorbar(avg_s[g], avg[g], xerr=sem_s[g], yerr=sem[g],
                color=style.getColor(g), fmt='s', markersize=3,
                markeredgewidth=mpl.rcParams['axes.linewidth'],
                markeredgecolor='k', ecolor='k',
                label={'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])

ax.plot([25,75],[25,75], ls=':', color='k', alpha=.5, zorder=-1)    

ax.set_xlim((25,75))
ax.set_ylim((25,75))
ax.set_xticks(np.arange(25,76,25))
ax.set_yticks(np.arange(25,76,25))
ax.set_aspect('equal')
ax.set_xlabel('expected')
ax.set_ylabel('observed')
ax.text(50, 75, 'μm to nearest\ntuned neighbor', ha='center', va='center',
        fontdict={'fontsize':7})
ax.legend(loc='lower right', bbox_to_anchor=(1.1, .05))
ax.set_aspect('equal')
sns.despine(ax=ax)

'''

#%%
speedTunings = analysisOpenField.calculateSpeedTuning(endoDataPath)
speedTunings["genotype"] = speedTunings.sess.str.split("_").str[0]
speedTunings["animal"] = speedTunings.sess.str.split("_").str[1]
speedTunings["date"] = speedTunings.sess.str.split("_").str[2]
speedTunings["speedTuned"] = speedTunings.pct > 0.995
runTuned = (tuningData.query("action == 'running'").set_index(["animal", "date", "neuron"]).pct > 0.995).rename("runTuned")
speedTunings = speedTunings.join(runTuned, on=["animal", "date", "neuron"])
for subset in ("all", "runTuned"):
    if subset == "all":
        gby = speedTunings.groupby("sess")
    else:
        gby = speedTunings[speedTunings.runTuned].groupby("sess")
    df = pd.DataFrame({'genotype': gby.genotype.first(),
                       'animal': gby.animal.first(),
                       'date': gby.date.first(),
                       'nNeurons': gby.size(),
                       'percTuned': gby.speedTuned.mean()*100})
    ax = layout.axes['speedTunings_'+subset]['axis']
    sessionBarPlot.sessionBarPlot(df, "percTuned", ax, style.getColor, weightCol="nNeurons")
    ax.set_ylim(0,100)
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    sns.despine(ax=ax)
    if subset == 'all':
        ax.set_title("all neurons", pad=2)
        ax.set_xticks([])
    else:
        ax.set_title("tuned to running", pad=2)
        ax.set_xticklabels(["D1", "A2A", "Oprm1"])

#%%

avgPerSpeed = analysisOpenField.avgActivityPerSpeed("data/endoData_2019.hdf")
avgPerSpeed.drop(columns=[-1], inplace=True)
avgPerSpeed -= avgPerSpeed[0][:,np.newaxis]
meta = avgPerSpeed.index.to_frame()
meta["genotype"], meta["animal"], meta["date"] = meta.session.str.split("_").str
bins = np.array([-.00001, .1, 1, 2, 3, 5, 8, 10])
xx = 0.5*(bins[1:]+bins[:-1])
for gt in ("d1", "a2a", "oprm1"):
    ax = layout.axes["activityPerSpeed_"+gt]["axis"]
    subset = avgPerSpeed[meta.genotype==gt]
    for (sess, nNeurons), r in subset.iterrows():
        ax.plot(xx, r, color=style.getColor(gt), lw=nNeurons/400.0)
    mean = [np.average(subset[col], weights=meta[meta.genotype==gt].noNeurons) for col in subset]
    ax.plot(xx, mean, lw=1.8, color=style.getColor(gt))
    ax.set_ylim(-0.01, 0.1)
    sns.despine(ax=ax)
    if gt=="a2a":
        ax.set_ylabel('Δsd')
    if gt=="oprm1":
        ax.set_xticks((0,5,10))
        ax.set_xlabel("velocity (cm/s)")
    else:
        ax.set_xticks([])
    ax.text(0.5, 0.08, longGtNames[gt])#, color=style.getColor(gt))
    #ax.set_title(gt)


#%%
ewindows = analysisOpenField.getEventWindows(endoDataPath, ["leftTurn", "rightTurn", "running"])
ewindows.set_index(["animal", "date", "neuron", "label"], inplace=True)
ewindows["tuned"] = tuningData.set_index(["animal", "date", "neuron", "action"]).pct>0.0995
ewindows.reset_index(inplace=True)

longerName = {'leftTurn': 'left turn', 'rightTurn': 'right turn', 'running': 'running'}

for label in ("leftTurn", "rightTurn", "running"):
    ax = layout.axes["onset_"+label]["axis"]
    for gt in ("d1", "a2a", "oprm1"):
        mask = np.logical_and(ewindows.genotype==gt, ewindows.label==label)
        mask = np.logical_and(mask, ewindows.tuned)
        ma = ewindows[mask].groupby(["animal", "date", "neuron"]).mean()["frameNo"]
        m = ma.mean()
        s = ma.sem()
        ax.plot(np.linspace(-1,1,40), m, color=style.getColor(gt))
        ax.fill_between(np.linspace(-1,1,40), m-s, m+s, color=style.getColor(gt), alpha=0.15)
    ax.axvline(0, color='k', linestyle='--', lw=0.75)
    ax.set_title(longerName[label] + ' tuned', pad=4)
    ax.set_xlabel("time from\n"+longerName[label]+" onset (s)")
    sns.despine(ax=ax)
layout.axes["onset_leftTurn"]["axis"].set_ylabel("mean activity (z-score)")
lines = [mpl.lines.Line2D([], [], color=style.getColor(gt)) for gt in ["d1", "a2a", "oprm1"]]
layout.axes["onset_running"]["axis"].legend(lines, ["D1", "A2A", "Oprm1"], bbox_to_anchor=(1.05, 0.4, 0.2, 0.2))
#layout.axes["onset_rightTurn"]["axis"].set_xlabel("time from onset (s)")

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
decodingData = analysisOpenField.decodingConfusion(endoDataPath, segmentedBehavior)
order = ["stationary", "running", "leftTurn", "rightTurn"]
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]
for gt, data in decodingData.groupby("genotype"):
    weightedData = data.set_index(["true", "predicted"]).eval("occurencies * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    ax = layout.axes["confusionMatrix_{}".format(gt)]["axis"]
    yticks = [behaviorNames[b] for b in order] if gt == "oprm1" else False
    sns.heatmap(weightedData[order].reindex(order), ax=ax, vmin=0, vmax=1, annot=True, fmt=".0%", cmap=cmocean.cm.amp,
                cbar=False, xticklabels=[behaviorNames[b] for b in order],
                yticklabels=yticks, annot_kws={'fontsize': 4.5},
                linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel("predicted" if gt=="d1" else None)
    ax.set_ylabel("truth" if gt=="oprm1" else None)
    ax.set_title(genotypeNames[gt])
    ax.set_ylim(4, 0)
    ax.tick_params("both", length=0, pad=3)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "openFieldNew.svg")
print("Done")
