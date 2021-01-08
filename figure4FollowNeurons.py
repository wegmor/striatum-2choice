import pathlib
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import figurefirst
import cmocean
import pims
import tqdm
import skimage
from utils import readSessions, fancyViz, alluvialPlot
from utils.sessionBarPlot import bootstrapSEM
import analysisOpenField
import analysisTunings
import analysisOftVs2Choice
import analysisKinematicsSupp
import style

style.set_context()
plt.ioff()

#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path('svg')
templateFolder = pathlib.Path('templates')
videoFolder = pathlib.Path('data')

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
svgName = 'figure4FollowNeurons.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()

#%%
genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn', 'none': 'untuned'}
phaseNames = {'mC2L': 'center-to-left', 'mC2R': 'center-to-right', 'mL2C': 'left-to-center',
              'mR2C': 'right-to-center', 'pC': 'center port', 'pL2C': 'left port',
              'pR2C': 'right port', 'none': 'untuned'}

#%% example session
ofSess =  next(readSessions.findSessions(endoDataPath, task='openField',
                                         animal='5308', date='190201'))
chSess = next(readSessions.findSessions(endoDataPath, task='2choice',
                                        animal=ofSess.meta.animal,
                                        date=ofSess.meta.date))
#%% load videos
open_field_video = pims.open(str(videoFolder / ofSess.meta.video) + ".avi")
tracking = ofSess.readTracking()

segmented = analysisOpenField.segmentAllOpenField().loc[str(ofSess)]
background = np.median([open_field_video.get_frame(i) for i in tqdm.trange(2000)], axis=0)

two_choice_video = pims.open(str(videoFolder / chSess.meta.video) + ".avi")
cmap = sns.cubehelix_palette(start=1.4, rot=.8*np.pi, light=.75, as_cmap=True)

#%% get example oft turn from example session
ex_action = (0,80)
start, stop = segmented.loc[ex_action][["startFrame", "stopFrame"]]
frame_ids = list(range(start, stop+1, 5))
frames = np.array([open_field_video.get_frame(i) for i in frame_ids])
coords = tracking.loc[start:stop]


#%% plot first oft frame
ax = layout.axes['trajectoryIllustration','openField']['axis']

ax.imshow(skimage.exposure.adjust_log(frames[0],1.3))
xy = coords.iloc[0]
ax.plot([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
        [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
        color='yellow', lw=mpl.rcParams['axes.linewidth'], zorder=1)
ax.scatter([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
           [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
           color='yellow', zorder=1, marker='.')
ax.set_xlim((120,709))
ax.set_ylim((590,0))
ax.fill([120, 300, 120, 120], [0, 0, 80, 0], color=cmap(0.0), alpha=.6)
ax.text(120+10, 10, "+0.00s", fontsize=6, va="top",
        color="w")#bbox=dict(facecolor=cmap(0.0), alpha=1.0))
wallCorners = ofSess.getWallCorners()
cm2px = (wallCorners.lowerRight.x - wallCorners.lowerLeft.x)/49
ax.plot([709-5*cm2px, 709], [600, 600], 'k', clip_on=False)
ax.axis('off')

for i in range(1, 6):
    ax = layout.axes['trajectoryIllustration','openField_sub{}'.format(i)]['axis']
    ax.imshow(skimage.exposure.adjust_log(frames[i],1.3))
    xy = coords.iloc[i*5]
    ax.plot([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
            [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
            color='yellow', lw=mpl.rcParams['axes.linewidth']/5, zorder=1)
    ax.scatter([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
               [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
               color='yellow', s=.2, zorder=1, marker='.')
    ax.set_xlim((120,709))
    ax.set_ylim((590,0))
    ax.axhspan(0, 190, color=cmap(i*5/(stop-start)), alpha=.6)
    ax.text((120+709)/2, 50, "+{:.2f}s".format(i*5/20.0), fontsize=6, va="top",
            ha="center", color="w")#))
    ax.axis('off')


#%% plot example oft turn trajectory
ax = layout.axes['trajectoryIllustration','turnTrajectory']['axis']

diff = (frames - background).mean(axis=-1)
alpha = ((np.clip(diff, -75, -40) + 40) / -35)
dx = coords.iloc[0].body.x - coords.iloc[0].tailBase.x
dy = coords.iloc[0].body.y - coords.iloc[0].tailBase.y
rot = np.rad2deg(np.arctan2(dy, dx))+90
t = ax.transData
tr = mpl.transforms.Affine2D().rotate_deg_around(coords.iloc[0].body.x,
                                                 coords.iloc[0].body.y, -rot) + t

for i,xy in coords.iterrows():
    ax.plot([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
            [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
            color='yellow', transform=tr, zorder=1, lw=mpl.rcParams['axes.linewidth'])

xlims = ax.get_xlim()
ylims = ax.get_ylim()


plt.sca(ax)
for i in range(len(frames)):
    frame = frames[i].mean(axis=-1) + i*255 - 255*len(frames)/2# + 256
    fancyViz.imshowWithAlpha(frame, .9*alpha[i], 255*len(frames)/2, cmap=cmap,
                             transform=tr, interpolation='antialiased')

ax.hlines(coords.iloc[0].body.y, coords.iloc[0].body.x-60, coords.iloc[0].body.x+60,
          ls=':', lw=mpl.rcParams['axes.linewidth'], color='k', alpha=.5, zorder=2,
          clip_on=False)
ax.vlines(coords.iloc[0].body.x, coords.iloc[0].body.y-60, coords.iloc[0].body.y+60,
          ls=':', lw=mpl.rcParams['axes.linewidth'], color='k', alpha=.5, zorder=2,
          clip_on=False)

ax.set_xlim(np.array(xlims) + [-40,40])
ax.set_ylim(np.array(ylims)[::-1] + [20,-40])
ax.axis('off')
#sat = 255*len(frames)/2
#mpl.colors.Normalize(-sat, sat)
cax = layout.axes['trajectoryIllustration','turnTrajectory_colorbar']['axis']
cb = plt.colorbar(mpl.cm.ScalarMappable(None, cmap), cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
for t in (0, 0.5, 1.0):
    text = '+{:.2f}s\n({:.0f}%)'.format(t*(stop-start)/20.0, t*100)
    cax.text(t, -0.5, text, ha='center', va='top', fontdict={'fontsize':6})
cax.text(0.5, -3.5, 'time (progess)', ha='center', va="top", fontdict={'fontsize':6})


#%% plot 2 choice frame
chTracking = chSess.readTracking()
n = 3933
frame = two_choice_video.get_frame(n)

ax = layout.axes['trajectoryIllustration','choice']['axis']
ax.set_ylim((750,0))
ax.set_xlim((55,800))
t = ax.transData
tr = mpl.transforms.Affine2D().rotate_deg_around(frame.shape[1]/2, frame.shape[0]/2, -90) + t

ax.imshow(skimage.exposure.adjust_gamma(frame, .7), transform=tr)
xy = chTracking.loc[n]
ax.plot([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
        [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
        color='yellow', lw=mpl.rcParams['axes.linewidth'], zorder=1,
        transform=tr)
ax.scatter([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
           [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
           color='yellow', zorder=1, marker='.', transform=tr)
wallCorners = chSess.getWallCorners()
cm2px = (wallCorners.lowerRight.x - wallCorners.lowerLeft.x)/15
ax.plot([800-5*cm2px, 800], [770, 770], 'k', clip_on=False)
ax.axis('off')


#%% plot example session turn trajectories & average
oftTracking = analysisOftVs2Choice.getSmoothedTracking(endoDataPath,
                                                       ofSess.meta.genotype, ofSess.meta.animal,
                                                       ofSess.meta.date, 'openField')
oftTracking['behavior'] = oftTracking.behavior.astype('str')
oftTracking = oftTracking.loc[oftTracking.behavior.isin(['leftTurn','rightTurn'])]
oftTracking = analysisOftVs2Choice.processTracking(oftTracking) # add progress, center and rotate

chTracking = analysisOftVs2Choice.getSmoothedTracking(endoDataPath,
                                                       ofSess.meta.genotype, ofSess.meta.animal,
                                                       ofSess.meta.date, '2choice')
chTracking['behavior'] = chTracking.behavior.str.slice(0,4)
chTracking = chTracking.loc[chTracking.behavior.isin(['mL2C','mR2C','mC2L','mC2R'])]
chTracking = analysisOftVs2Choice.processTracking(chTracking)
tracking = pd.concat([oftTracking, chTracking], keys=['oft','2ch'], names=['task'])


#%%
ax = layout.axes['turnTrajectories']['axis']

for (task,behavior), bdf in tracking.groupby(['task','behavior']):
    mean = bdf.groupby('bin').mean()
    offsets = ({'leftTurn':-2, 'rightTurn':2, 'mR2C':-10, 'mC2L':-2, 'mC2R':2, 'mL2C':10}[behavior],
               {'oft':11, '2ch':-9}[task])
    for an, track in bdf.groupby('actionNo'):
        analysisOftVs2Choice.plotTrackingEvent(track, ax, alpha=.05, color=style.getColor(behavior),
                                               offsets=offsets, lw=mpl.rcParams['axes.linewidth'])
    analysisOftVs2Choice.plotTrackingEvent(mean, ax, offsets=offsets)

ax.text(0, 22, 'open field', color='k', fontsize=7, va='center', ha='center')
ax.text(-2, 21, 'left turn', color=style.getColor('leftTurn'), fontsize=6, va='top', ha='right')
ax.text(2, 21, 'right turn', color=style.getColor('rightTurn'), fontsize=6, va='top', ha='left')
ax.text(0, 1, '2-choice', color='k', fontsize=7, va='center', ha='center')
ax.text(-2, 0, 'center to\nleft turn', color=style.getColor('mC2L'),
        fontsize=6, va='top', ha='right')
ax.text(2, 0, 'center to\nright turn', color=style.getColor('mC2R'),
        fontsize=6, va='top', ha='left')
ax.text(-10, 0, 'right to\ncenter turn', color=style.getColor('mR2C'),
        fontsize=6, va='top', ha='right')
ax.text(10, 0, 'left to\ncenter turn', color=style.getColor('mL2C'),
        fontsize=6, va='top', ha='left')
ax.hlines(-15, -2.5, 2.5, color='k', lw=mpl.rcParams['axes.linewidth'], clip_on=False)
ax.text(0, -15.2, '5cm', ha='center', va='top', fontsize=6)
ax.set_xlim((-17,17))
ax.set_ylim((-15,23))
ax.set_aspect('equal')
ax.axis('off')


#%% kinematic paramters density plots
ofSegs = analysisKinematicsSupp.openFieldSegmentKinematics(endoDataPath)
tcSegs = analysisKinematicsSupp.twoChoiceSegmentKinematics(endoDataPath)
fps = 20
#Change from per frame to per second
tcSegs['bodyAngleSpeed'] *= -180/np.pi * fps
tcSegs['speed'] *= fps
ofSegs['bodyAngleSpeed'] *= -180/np.pi * fps
ofSegs['speed'] *= fps
phases = ['leftTurn', 'rightTurn', 'mC2L','mC2R','mL2C','mR2C']
for p in phases:
    ax = layout.axes['turnSpeedHist_'+p]['axis']
    if p.endswith("Turn"):
        data = ofSegs[ofSegs.label==p].bodyAngleSpeed
    else:
        data = tcSegs[tcSegs.label==p].bodyAngleSpeed
    sns.kdeplot(data, cut=0, ax=ax, color=style.getColor(p),
                alpha=1, shade=True, legend=False)
    ax.set_xlim(-150, 150)
    ax.set_ylim(0, 0.04)
    ax.set_xticks([-150,0,150])
    ax.set_yticks([])
    title = ("open field "+behaviorNames[p]) if p.endswith("Turn") else ("2-choice "+phaseNames[p])
    ax.text(0, 0.022, title, fontsize=6, color=style.getColor(p), ha="center")
    if p == "mL2C":
        ax.set_xlabel("turning speed (deg/s)")
    else:
        ax.set_xticklabels([])
    sns.despine(ax=ax, left=True, trim=True)

#%% remapping example neurons plot
ofTraces = ofSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
chTraces = chSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
ofTracking = analysisOftVs2Choice.getSmoothedTracking(endoDataPath, ofSess.meta.genotype,
                                                      ofSess.meta.animal, ofSess.meta.date,
                                                      'openField')
chTracking = analysisOftVs2Choice.getSmoothedTracking(endoDataPath, ofSess.meta.genotype,
                                                      ofSess.meta.animal, ofSess.meta.date,
                                                      '2choice')

for n,neuron in enumerate([86,192][::-1]):
    ofTrace = ofTraces[neuron]
    chTrace = chTraces[neuron]

    ### open field
    # schematic
    schemAx = layout.axes[('n{}'.format(n),'ofSchematic')]['axis']
    fig = schemAx.get_figure()
    fig.sca(schemAx)
    fv = fancyViz.OpenFieldSchematicPlot(ofSess, linewidth=mpl.rcParams['axes.linewidth'],
                                         smoothing=3, saturation=1)
    img = fv.draw(ofTrace, ax=schemAx)

    # top events
    topTenEventsAx = layout.axes[('n{}'.format(n),'ofTopTen')]['axis']
    topTenTracesAxOf = layout.axes[('n{}'.format(n),'ofTopTenTraces')]['axis']
    pIdx = analysisOftVs2Choice.plotTop10Events(ofTrace, ofTracking,
                                                axs=[topTenEventsAx, topTenTracesAxOf],
                                                offset=10)
    topTenEventsAx.vlines(-7, -15, -10, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenEventsAx.hlines(-15, -7, -2, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxOf.vlines(-2.5, -5, 5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxOf.hlines(-5, -2.5, 2.5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)

    # map
    mapAx = layout.axes[('n{}'.format(n),'ofMap')]['axis']
    fig.sca(mapAx)
    fv = fancyViz.TrackingIntensityPlot(session=ofSess, smoothing=20, saturation=1,
                                        drawBg=False)
    fv.draw(ofTrace, ax=mapAx)
    headCoords = fv.coordinates
    mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1], marker='x',
                  linewidth=mpl.rcParams['axes.linewidth'],
                  c=cmocean.cm.phase(np.arange(10)/11))

    ### 2-choice
    # schematic
    schemAx = layout.axes[('n{}'.format(n),'chSchematic')]['axis']
    fig.sca(schemAx)
    fv = fancyViz.SchematicIntensityPlot(chSess, linewidth=mpl.rcParams['axes.linewidth'],
                                         smoothing=5, splitReturns=False, saturation=1)
    img = fv.draw(chTrace, ax=schemAx)

    # top events
    topTenEventsAx = layout.axes[('n{}'.format(n),'chTopTen')]['axis']
    topTenTracesAxCh = layout.axes[('n{}'.format(n),'chTopTenTraces')]['axis']
    pIdx = analysisOftVs2Choice.plotTop10Events(chTrace, chTracking, 
                                                axs=[topTenEventsAx, topTenTracesAxCh],
                                                offset=10)
    topTenEventsAx.vlines(-7, -15, -10, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenEventsAx.hlines(-15, -7, -2, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxCh.vlines(-2.5, -5, 5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxCh.hlines(-5, -2.5, 2.5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)

    # map
    mapAx = layout.axes[('n{}'.format(n),'chMap')]['axis']
    fig.sca(mapAx)
    fv = fancyViz.TrackingIntensityPlot(session=chSess, smoothing=20, saturation=1,
                                        portsUp=True, drawBg=False)
    fv.draw(chTrace, ax=mapAx)
    headCoords = fv.coordinates
    mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1],
                  marker='x', linewidth=mpl.rcParams['axes.linewidth'],
                  c=cmocean.cm.phase(np.arange(10)/11),
                  transform=fv.transform)

    traceAxLims = np.concatenate([topTenTracesAxCh.get_ylim(),topTenTracesAxOf.get_ylim()])
    traceAxLims = (traceAxLims.min(), traceAxLims.max())
    topTenTracesAxCh.set_ylim(traceAxLims)
    topTenTracesAxOf.set_ylim(traceAxLims)

cax = layout.axes['second_colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})


#%%
openFieldTunings = analysisOpenField.getTuningData(endoDataPath)
twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)
for t in (twoChoiceTunings, openFieldTunings):
    t['signp'] = t['pct'] > .995
    t.set_index(["animal", "date", "action", "neuron"], inplace=True)
    t.sort_index(inplace=True)

labels = ["leftTurn"]#, "mL2C", "mR2C", "all", "running", "mC2L", "mC2R"
saturation = 0.3
fvs = {}
for key in itertools.product(('d1', 'a2a', 'oprm1'), labels):
    fvs[("2choice",)+key] = fancyViz.SchematicIntensityPlot(splitReturns=False,
                                linewidth=mpl.rcParams['axes.linewidth'],
                                smoothing=5, saturation=saturation)

for key in itertools.product(('d1', 'a2a', 'oprm1'), labels):
    fvs[("openField",)+key] = fancyViz.OpenFieldSchematicPlot(
                                linewidth=mpl.rcParams['axes.linewidth'],
                                smoothing=3, saturation=saturation)

for sess in readSessions.findSessions(endoDataPath, task=["2choice", "openField"]):
    shortKey = (sess.meta.animal, sess.meta.date)
    if shortKey not in openFieldTunings.index:
        continue
    if shortKey not in twoChoiceTunings.index:
        continue
    traces = sess.readDeconvolvedTraces(zScore=True)
    genotype = sess.meta.genotype
    task = sess.meta.task
    for label in labels:
        fvs[(task, genotype, label)].setSession(sess)
        if label == "all": #Don't look at tuning
            tuned = np.full(traces.shape[1], True)
        elif label[0] == 'm': #2-choice tuning
            tuned = twoChoiceTunings.loc[shortKey+(label+"-",)].signp
        else: #Open field tuning
            tuned = openFieldTunings.loc[shortKey+(label,)].signp
        for neuron in traces.columns:
            if tuned[neuron]:
                fvs[(task, genotype, label)].addTraceToBuffer(traces[neuron])
for task, gt, label in itertools.product(("openField", "2choice"),
                                         ("d1", "a2a", "oprm1"), labels):
    axName = "_".join(("mean", label, "of" if task=="openField" else task, gt))
    img = fvs[(task, gt, label)].drawBuffer(ax=layout.axes[axName]['axis'])

for i in range(1,2):#4):
    cax = layout.axes['colorbar_means_{}'.format(i)]['axis']
    cb = plt.colorbar(img, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cax.set_axis_off()
    cax.text(-0.325, -.1, "{:.1f}".format(-saturation), ha='right', va='center', fontsize=6)
    cax.text(0.325, -.1, "{:.1f}".format(saturation), ha='left', va='center', fontsize=6)
    cax.text(0, 0.5, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})


#%% oft vs choice left/right tuning comparison
ofTuningData = analysisOpenField.getTuningData(endoDataPath)
chTuningData = analysisTunings.getTuningData(endoDataPath)

tuningData = (pd.concat([chTuningData, ofTuningData], keys=['choice','oft'], names=['task'])
                .reset_index('task').reset_index(drop=True))
tuningData['signp'] = tuningData.pct > .995
tuningData = tuningData.set_index(['task','action','genotype','animal','date','neuron'])

signTuned = tuningData.unstack(['task','action']).dropna()['signp']
#signTuned[('choice','none')] = signTuned.choice.sum(axis=1) == 0
#signTuned[('choice','port')] = signTuned.choice.loc[:,signTuned.choice.columns.str.contains('^p|^d')].any(axis=1)
signTuned[('choice','leftTurn')] = signTuned.choice[['mC2L-','mR2C-']].any(axis=1)
signTuned[('choice','rightTurn')] = signTuned.choice[['mC2R-','mC2R-']].any(axis=1)

for tuning in ['leftTurn', 'rightTurn']:
    data = signTuned.loc[:,signTuned.columns.get_level_values(1) == tuning].droplevel(1, axis=1)

    noNeurons = data.groupby(['genotype','animal','date']).size()
    pctTund = data.astype('int').groupby(['genotype','animal','date']).sum() / noNeurons.values[:,np.newaxis]
    pctTund *= 100
    
    obsOvlp = ((data.oft * data.choice).groupby(['genotype','animal','date']).sum() / \
                noNeurons)
    expOvlp = ((data.oft.groupby(['genotype','animal','date']).sum() / \
                noNeurons) * \
               (data.choice.groupby(['genotype','animal','date']).sum() / \
                noNeurons))
    ovlpDf = pd.concat([obsOvlp, expOvlp], axis=1, keys=['observed','expected']) * 100

    axs = [layout.axes['{}PctTuned'.format(tuning)]['axis'], 
           layout.axes['{}Ovlp'.format(tuning)]['axis']]

    for gt,df in pctTund.groupby('genotype'):
        x = {'d1':0,'a2a':1,'oprm1':2}[gt]
        ax = axs[0]
        means = np.average(df[['oft','choice']], axis=0, weights=noNeurons.loc[gt])
        sems = [bootstrapSEM(df.oft, noNeurons.loc[gt]),
                bootstrapSEM(df.choice, noNeurons.loc[gt])]
        ax.errorbar(x=x-.2, y=means[0], yerr=sems[0],
                    marker='^', color=style.getColor(gt), clip_on=False)
        ax.errorbar(x=x+.2, y=means[1], yerr=sems[1],
                     marker='s', color=style.getColor(gt), clip_on=False)
        ax.plot([[x-.2]*len(df),[x+.2]*len(df)], df.T.values,
                color=style.getColor(gt), alpha=.25, zorder=-99, clip_on=False)
        ax.axhline(0, ls=':', alpha=.5, color='k', zorder=-100,
                   lw=mpl.rcParams['axes.linewidth'])

    for gt,df in ovlpDf.groupby('genotype'):
        x = {'d1':0,'a2a':1,'oprm1':2}[gt]
        ax = axs[1]
        means = np.average(df[['observed','expected']], axis=0, weights=noNeurons.loc[gt])
        sems = [bootstrapSEM(df.observed, noNeurons.loc[gt]),
                bootstrapSEM(df.expected, noNeurons.loc[gt])]
        ax.errorbar(x=x-.2, y=means[0], yerr=sems[0],
                    marker='o', color=style.getColor(gt), clip_on=False)
        ax.errorbar(x=x+.2, y=means[1], yerr=sems[1],
                    marker='o', markeredgecolor=style.getColor(gt),
                    markerfacecolor='w', ecolor=style.getColor(gt), clip_on=False)
        ax.plot([[x-.2]*len(df),[x+.2]*len(df)], df.T.values,
                color=style.getColor(gt), alpha=.25, zorder=-99, clip_on=False)
        ax.axhline(0, ls=':', alpha=.5, color='k', zorder=-100,
                   lw=mpl.rcParams['axes.linewidth'])

    for ax in axs:
        ax.set_ylim(0,100)
        ax.set_yticks([0,50,100])
        ax.set_yticks([25,75], minor=True)
        ax.set_xlim((-.5,2.5))
        ax.set_xticks([])
        sns.despine(ax=ax, bottom=True)
        if tuning == 'rightTurn': ax.set_yticklabels([])
    if tuning == 'leftTurn':
        legend_elements = [mpl.lines.Line2D([0], [0], marker='^', ls='', color='k',
                                            label='open field'),
                           mpl.lines.Line2D([0], [0], marker='s', ls='', color='k',
                                            label='2-choice')]
        axs[0].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(.5,.95))
        axs[0].set_ylabel('% tuned', labelpad=0)
        legend_elements = [mpl.lines.Line2D([0], [0], marker='o', ls='', color='k',
                                            label='observed'),
                           mpl.lines.Line2D([0], [0], marker='o', ls='', markeredgecolor='k',
                                            markerfacecolor='w', label='expected')]
        axs[1].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(.5,.95))
        axs[1].set_ylabel('% overlap', labelpad=0)
        axs[0].set_title('left turn')
    else:
        axs[0].set_title('right turn')

axt = layout.axes['gt_legend']['axis']
legend_elements = [mpl.patches.Patch(color=style.getColor(gt), alpha=1,
                                     label=genotypeNames[gt])
                   for gt in ['d1','a2a','oprm1']
                  ]
axt.legend(handles=legend_elements, ncol=3, loc='center')
axt.axis('off')


#%% alluvial plot
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
openFieldTunings = analysisOpenField.getTuningData(endoDataPath, segmentedBehavior)

twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)

for t in (twoChoiceTunings, openFieldTunings):
    t['signp'] = t['pct'] > .995
    t['signn'] = t['pct'] < .005

primaryTwoChoice = twoChoiceTunings.loc[twoChoiceTunings.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
primaryTwoChoice.loc[~primaryTwoChoice.signp, 'action'] = 'none'
primaryOpenField = openFieldTunings.loc[openFieldTunings.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
primaryOpenField.loc[~primaryOpenField.signp, 'action'] = 'none'

primaryPairs = primaryOpenField.join(primaryTwoChoice.set_index(["animal", "date", "neuron"]),
                                     on=["animal", "date", "neuron"], rsuffix="_2choice", how="inner")
primaryPairs = primaryPairs[["genotype", "animal", "date", "neuron", "action", "action_2choice"]]
primaryPairs.rename(columns={'action': 'action_openField'}, inplace=True)

order_openField = ["stationary", "running", "leftTurn", "rightTurn", "none"]
order_twoChoice = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "dL2C-", "pL2Co", "pL2Cr",
                   "pC2L-", "pC2R-", "dR2C-", "pR2Co", "pR2Cr", "none"]
primaryPairs.action_2choice = pd.Categorical(primaryPairs.action_2choice, order_twoChoice)
primaryPairs.action_openField = pd.Categorical(primaryPairs.action_openField, order_openField)

colormap = {a: style.getColor(a[:4]) for a in primaryPairs.action_2choice.unique()}
colormap.update({a: style.getColor(a) for a in primaryPairs.action_openField.unique()})


for gt in ("d1", "a2a", "oprm1"):
    ax = layout.axes['alluvial_{}'.format(gt)]['axis']
    data = primaryPairs.query("genotype == '{}'".format(gt))
    leftY, rightY = alluvialPlot.alluvialPlot(data, "action_openField",
                                              "action_2choice", colormap, ax, 
                                              colorByRight=False, alpha=0.75)
    ax.set_xlim(-0.2,1.2)
    ax.axis("off")
    ax.set_title(genotypeNames[gt])
    if gt=="d1":
        ticks = leftY.groupby(level=0).agg({'lower': np.min, 'higher': np.max}).mean(axis=1)
        for label, y in ticks.items():
            ax.text(-0.25, y, behaviorNames[label], ha="right", va="center",
                    color=style.getColor(label))
    if gt=="oprm1":
        newLabels = rightY.index.get_level_values(1).str[:4]
        newLabels = newLabels.str.replace('pC2.', 'pC')
        newLabels = newLabels.str.replace('d', 'p')
        ticks = rightY.groupby(newLabels).agg({'lower': np.min, 'higher': np.max}).mean(axis=1)
        for label, y in ticks.items():
            ax.text(1.25, y, phaseNames[label], ha="left", va="center",
                    color=style.getColor(label))
print("alluvial (followed neurons):")
print(primaryPairs.groupby("genotype").size())


#%%
kinematics_of_all = analysisKinematicsSupp.openFieldSegmentKinematics(endoDataPath)
kinematics_tc_all = analysisKinematicsSupp.twoChoiceSegmentKinematics(endoDataPath)
fps = 20
#Change from per frame to per second
kinematics_of_all['bodyAngleSpeed'] *= -180/np.pi * fps
kinematics_of_all['speed'] *= fps
kinematics_tc_all['bodyAngleSpeed'] *= -180/np.pi * fps
kinematics_tc_all['speed'] *= fps

ax = layout.axes['kinematics3d_openField']['axis']
ax.view_init(60, -70)#(30, -60)
ax.scatter(kinematics_of_all.bodyAngleSpeed,
           kinematics_of_all.speed,
           kinematics_of_all.elongation, s=.5,
           c=[style.getColor(l) for l in kinematics_of_all.label],
           alpha=.2, lw=0, rasterized=True)
ax.set_title("open field")

ax = layout.axes['kinematics3d_2choice']['axis']
ax.view_init(60, -70)#(30, -60)
validPhases = ["mC2L", "mC2R", "mL2C", "mR2C",
               "pC2L", "pC2R", "dL2C", "dR2C",
               "pL2C", "pR2C"]
mask = kinematics_tc_all.label.isin(validPhases)
ax.scatter(kinematics_tc_all.bodyAngleSpeed[mask],
           kinematics_tc_all.speed[mask],
           kinematics_tc_all.elongation[mask], s=.5,
           c=[style.getColor(l) for l in kinematics_tc_all.label[mask]],
           alpha=.2, lw=0, rasterized=True)
ax.set_title("2-choice")

for ax in (layout.axes['kinematics3d_openField']['axis'],
           layout.axes['kinematics3d_2choice']['axis']):
    ax.set_xlim(-120, 120)
    ax.set_ylim(-7.5, 25)
    ax.set_zlim(2.5, 4)
    ax.set_zticks(np.arange(2.5, 4.5, 0.5))
    ax.tick_params(pad=-3.5)
    ax.set_xlabel("turning speed (deg/s)", labelpad=-8)
    ax.set_ylabel("speed (cm/s)", labelpad=-8)
    ax.set_zlabel("elongation (cm)", labelpad=-8)

#%%
nNeurons = []
for sess in analysisKinematicsSupp.find2choiceSessionsFollowingOpenField(endoDataPath):
    deconv = sess.readDeconvolvedTraces(zScore=False)
    nNeurons.append((str(sess), deconv.shape[1]))
nNeurons = pd.DataFrame(nNeurons, columns=["sess", "nNeurons"])
nNeurons.set_index("sess", inplace=True)

twoChoicePdists = analysisKinematicsSupp.twoChoicePdists(endoDataPath)
openFieldPdists = analysisKinematicsSupp.openFieldPdists(endoDataPath)
cdists = analysisKinematicsSupp.openFieldToTwoChoiceCdists(endoDataPath)
dist_list = [twoChoicePdists, openFieldPdists, cdists]
cols = ["C3", "C4", "C5"]
gts = ["d1", "a2a", "oprm1"]
for i, dist in enumerate(dist_list):
    mean = {gt: np.zeros(20-1) for gt in gts}
    nTot = {gt: 0 for gt in gts}
    nPairs = {gt: np.zeros(20-1) for gt in gts}
    for s, g in dist.groupby(level=0):
        bins = pd.cut(g.kinematics_dist, np.linspace(.1, 4, 20))
        binned = g.groupby(bins).mean()
        gt = s.split("_")[0]
        n = nNeurons.loc[s].nNeurons
        ax = layout.axes['kinematicsVsDeconv_'+gt]['axis']
        ax.plot(binned.kinematics_dist, binned.deconv_dist, color=cols[i],
                alpha=np.clip(n/500, 0.1, 1.0), lw=.5)
        mean[gt] += n*binned.deconv_dist
        nPairs[gt] += g.groupby(bins).size()
        nTot[gt] += n
    for gt in gts:
        mean[gt] /= nTot[gt]
        ax = layout.axes['kinematicsVsDeconv_'+gt]['axis']
        ax.plot(binned.kinematics_dist, mean[gt], color=cols[i])#, lw=2)
        ax = layout.axes['kinematicsPairHist_'+gt]['axis']
        nPairs[gt] /= nPairs[gt].sum()
        ax.plot(binned.kinematics_dist, nPairs[gt]*4, color=cols[i])

gt_names = {"d1": "D1+", "a2a": "A2A+", "oprm1": "Oprm1+"}
for gt in gts:
    ax = layout.axes['kinematicsVsDeconv_'+gt]['axis']
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.05, 0.12)
    ax.axhline(0, color="k", alpha=0.3, lw=0.5, linestyle="--")
    ax.set_title(gt_names[gt])#, color=style.getColor(gt))
    ax.set_yticks(np.arange(-0.05, 0.15, 0.05))
    ax.set_xticklabels([])
    if gt=="d1":
        ax.set_ylabel("ensamble correlation")
    else:
        ax.set_yticklabels([])
    #if gt=="a2a":
    #    #'center right')
    sns.despine(ax=ax)

    ax = layout.axes['kinematicsPairHist_'+gt]['axis']
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 0.5)
    if gt=="d1":
        ax.set_ylabel("pdf")
    else:
        ax.set_yticklabels([])
    if gt=="a2a":
        ax.set_xlabel("kinematic dissimilarity (Mahalanobis distance)")
        lines = [mpl.lines.Line2D([], [], color=c, label=l)
                 for c,l in zip(cols, ["open field → open field",
                                       "2-choice → 2-choice",
                                       "open field → 2-choice"])]
        ax.legend(handles=lines, ncol=3, mode="expand",
                  bbox_to_anchor=(-1.6, -1.2, 3.8, .1))
    sns.despine(ax=ax)

#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)
#subprocess.check_call(['inkscape', '-f', outputFolder / svgName,
#                                   '-A', outputFolder / (svgName[:-3]+'pdf')])
