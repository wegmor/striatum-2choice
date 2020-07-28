#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:06:48 2020

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import cmocean
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from utils import readSessions, fancyViz, particleFilter, segmentBehaviors
from utils.cachedDataFrame import cachedDataFrame
import analysisOpenField, analysisTunings
import pathlib
import figurefirst
#import av
#from skimage.color import rgb2gray
import style
from matplotlib.backends.backend_pdf import PdfPages
plt.ioff()

#%%
style.set_context()
endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()


#%%
def getSmoothedTracking(dataFile, genotype, animal, date, task):
    def _medSmoothNonBody(tracking):
        tracking = tracking.copy()
        for col in ['leftEar','rightEar','tailBase']:
            _tracking = tracking[col].copy()
            _tracking.loc[_tracking.likelihood<.9,['x','y']] = np.nan
            _tracking.loc[:,['x','y']] = (_tracking[['x','y']].rolling(11, center=True, min_periods=5)
                                                              .median())
            tracking.loc[:,col] = _tracking.values
        return tracking
    @cachedDataFrame('smoothed_{}_tracking_{}-{}-{}.pkl'.format(task, genotype, animal, date))
    def _smoothTracking():
        sess = next(readSessions.findSessions(dataFile, task=task, genotype=genotype,
                                              animal=animal, date=date))
        tracking = sess.readTracking(inCm=True)
        tracking = _medSmoothNonBody(tracking)
        coords = particleFilter.particleFilter(tracking, flattening = 1e-12)
        colIdx = ((tracking.columns.get_level_values(0) == 'body') & \
                   tracking.columns.get_level_values(1).isin(['x','y']))
        tracking.loc[:,colIdx] = coords[['x','y']].values
        coords.rename(columns={"bodyAngle": "bodyDirection"}, inplace=True)
        coords.rename_axis("time", axis=0, inplace=True)
        coords.bodyDirection = np.rad2deg(coords.bodyDirection)
        if task == 'openField':
            behaviors = segmentBehaviors.segmentBehaviors(coords)[['startFrame','behavior']]
            behaviors.insert(0, "actionNo", behaviors.index.copy())
            coords.reset_index(inplace=True)
            coords.rename(columns={'time':'frame'}, inplace=True)
            coords = coords.merge(behaviors, left_on='frame', right_on='startFrame',
                                  how='left').fillna(method='ffill')
            tracking['actionNo'] = coords.actionNo
            #tracking['actionProgress'] = coords.actionProgress
            tracking['behavior'] = coords.behavior
        else:
            behaviors = sess.labelFrameActions(reward='fullTrial', switch=True).reset_index(drop=True)
            behaviors.index.name = 'frame'
            coords.reset_index(inplace=True)
            coords.rename(columns={'time':'frame'}, inplace=True)
            coords = coords.merge(behaviors, left_on='frame', right_on='frame', how='left')
            tracking['actionNo'] = coords.actionNo
            tracking['actionProgress'] = coords.actionProgress
            tracking['behavior'] = coords.label
        return tracking
    return _smoothTracking()

def getBodyAngle(bodyXY, tailBaseXY):
    xy = bodyXY - tailBaseXY
    bodyAngle = np.arctan2(xy[:,1],xy[:,0])
    return bodyAngle

# https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python#34374437
def rotate(p, angle=0, origin=(0, 0)):
    #angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def getTop10Events(trace, framesBefore=5, framesAfter=15):
    # max events at least 1 sec separated from other included events
    pIdx, pProp = find_peaks(trace, distance=max(framesBefore,framesAfter), width=1)
    # find top 10 most prominent (not necessarily highest!) events
    sortIdx = np.argsort(pProp['prominences'])
    pIdx = pIdx[sortIdx]
    pIdx = pIdx[(pIdx >= framesBefore) & (pIdx <= pIdx.max() - framesAfter)][-10:]
    return np.array(pIdx)[::-1]

def plotTopEvents(trace, tracking, pIdx, axs=None, framesBefore=5, framesAfter=15, offset=20):
    trace = trace.copy()
    tracking = tracking.copy()
    
    if not axs:
        fig, axs = plt.subplots(2, 1, figsize=(2.5, 1.25))
    ax, tax = axs
    
    tlen = framesBefore+framesAfter+1
    traceOffset = 1.25 * tlen

    for i,idx in enumerate(pIdx):
        trackAll = tracking.loc[idx-framesBefore:idx+framesAfter].copy()
        body = trackAll['body'][['x','y']]
        origin = body.iloc[0]
        body -= origin
        head = (trackAll['leftEar'][['x','y']] + trackAll['rightEar'][['x','y']]) / 2
        head -= origin
        tail = trackAll['tailBase'][['x','y']]
        tail -= origin
        
        angle = np.arctan2(tail.iloc[0]['x'], tail.iloc[0]['y']) + np.deg2rad(180)
        body.loc[:,['x','y']] = rotate(body, angle=angle)
        head.loc[:,['x','y']] = rotate(head, angle=angle)
        tail.loc[:,['x','y']] = rotate(tail, angle=angle)
        
        color = cmocean.cm.phase(i/11)
        
        t = trace.loc[idx-framesBefore:idx+framesAfter].values
        x = np.arange(i*traceOffset, i*traceOffset+tlen) - tlen/2
        tax.vlines(x, 0, t, color=color, clip_on=False)
        
        ax.hlines([0], i*offset-offset*.4, i*offset+offset*.4,
                  zorder=99, alpha=1, ls=':', lw=mpl.rcParams['axes.linewidth'])
        ax.vlines([i*offset], -offset*.4, offset*.4, 
                  zorder=99, alpha=1, ls=':', lw=mpl.rcParams['axes.linewidth'])
        for n,f in enumerate(range(-framesBefore,framesAfter+1)):
            ax.plot(np.stack([head.loc[idx+f,'x']+i*offset,
                              body.loc[idx+f,'x']+i*offset, 
                              tail.loc[idx+f,'x']+i*offset]),
                    np.stack([head.loc[idx+f,'y'], body.loc[idx+f,'y'], tail.loc[idx+f,'y']]),
                    color=color, alpha=np.clip(((tlen-n)/tlen)**1.5+.25,0,1),
                    zorder=-99, clip_on=False)
#            ax.scatter(head.loc[idx+n,'x']+i*offset, head.loc[idx+n,'y'],
#                       marker='x', s=5, color=color, alpha=1-((n+5)/30)**.5, clip_on=False)
    
    ax.set_aspect('equal')
    ax.set_xlim(-offset, i*offset+offset)
    ax.set_ylim(-.6*offset, .6*offset)
    tax.set_xlim([-traceOffset, i*traceOffset+traceOffset])
    ax.axis('off')
    tax.axis('off')

def plotTrackingEvent(track, ax, color='k', alpha=1., rotateAngle='auto', subtractOrigin=True):
    body = track['body'][['x','y']]
    head = (track['leftEar'][['x','y']] + track['rightEar'][['x','y']]) / 2
    tail = track['tailBase'][['x','y']]
    if subtractOrigin:
        origin = body.iloc[0]
        body -= origin
        head -= origin
        tail -= origin
    
    if rotateAngle == 'auto':
        angle = np.arctan2(tail.iloc[0]['x'], tail.iloc[0]['y']) + np.deg2rad(180)
        body.loc[:,['x','y']] = rotate(body, angle=angle)
        head.loc[:,['x','y']] = rotate(head, angle=angle)
        tail.loc[:,['x','y']] = rotate(tail, angle=angle)
    elif isinstance(rotateAngle, int):
        body.loc[:,['x','y']] = rotate(body, angle=np.deg2rad(rotateAngle))
        head.loc[:,['x','y']] = rotate(head, angle=np.deg2rad(rotateAngle))
        tail.loc[:,['x','y']] = rotate(tail, angle=np.deg2rad(rotateAngle))
    
    for n in range(len(head)):
        ax.plot(np.stack([head.iloc[n].x,
                          body.iloc[n].x, 
                          tail.iloc[n].x]),
                np.stack([head.iloc[n].y, body.iloc[n].y, tail.iloc[n].y]),
                color=color, alpha=np.clip(((len(head)-n)/len(head))**1.5+.25,0,1)*alpha,
                clip_on=False)


#%%
ofTuningData = analysisOpenField.getTuningData(endoDataPath)
chTuningData = analysisTunings.getTuningData(endoDataPath)

    
#%%
#svgName = 'oftChoiceComp.svg'
#layout = figurefirst.FigureLayout(templateFolder / svgName)
#layout.make_mplfigures()

selTask = 'oft'
for ofSess in readSessions.findSessions(endoDataPath, task='openField',
                                        filterQuery='date != "190224"'):
    pdf = PdfPages('svg/oftChoiceComp/{}_{}_{}_{}-sorted.pdf'.format(ofSess.meta.genotype,
                                                                     ofSess.meta.animal,
                                                                     ofSess.meta.date,
                                                                     selTask))
    
    chSess = next(readSessions.findSessions(endoDataPath, task='2choice',
                                            genotype=ofSess.meta.genotype,
                                            animal=ofSess.meta.animal,
                                            date=ofSess.meta.date))
    
    ofTraces = ofSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    chTraces = chSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    #ofTracking = ofSess.readTracking(inCm=True).reset_index(drop=True)
    ofTracking = getSmoothedTracking(endoDataPath, ofSess.meta.genotype,
                                     ofSess.meta.animal, ofSess.meta.date,
                                     'openField')
    #chTracking = chSess.readTracking(inCm=True).reset_index(drop=True)
    chTracking = getSmoothedTracking(endoDataPath, ofSess.meta.genotype,
                                     ofSess.meta.animal, ofSess.meta.date,
                                     '2choice')

    ofTuning = ofTuningData.query('animal == @ofSess.meta.animal & date == @ofSess.meta.date')
    chTuning = chTuningData.query('animal == @ofSess.meta.animal & date == @ofSess.meta.date')
    tuning = pd.concat([chTuning, ofTuning], keys=['choice','oft'], names=['task']).reset_index(level=0)
    #neurons = (tuning.groupby('neuron').tuning.max()
    #                 .sort_values(ascending=False).index.astype('int').values)
    
    neurons = (tuning.loc[tuning.groupby(['task','neuron']).tuning.idxmax().values]
                     .sort_values(['task','tuning'], ascending=False).set_index(['task','action']))
    neurons = neurons.loc[selTask].neuron.values.astype('int')
    
    # 278, 43, 166, 86, 117, 77, 0, 224, 217, 198, 73, 97 <- 5308, 190201
    for n,neuron in enumerate(neurons[:30]):
        svgName = 'oftChoiceCompSmall.svg'
        layout = figurefirst.FigureLayout(templateFolder / svgName)
        layout.make_mplfigures()
        n = 1 #n += 1
        ofTrace = ofTraces[neuron]
        chTrace = chTraces[neuron]
    
        # open field
        # schematic
        schemAx = layout.axes[('n{}'.format(n),'ofSchematic')]['axis']
        fig = schemAx.get_figure()
        fig.sca(schemAx)
        fv = fancyViz.OpenFieldSchematicPlot(ofSess, linewidth=mpl.rcParams['axes.linewidth'],
                                             smoothing=3, saturation=1)
        img = fv.draw(ofTrace, ax=schemAx)
        
        # top events
        topTenEventsAx = layout.axes[('n{}'.format(n),'ofTopTen')]['axis']
        topTenTracesAx = layout.axes[('n{}'.format(n),'ofTopTenTraces')]['axis']
        pIdx = getTop10Events(ofTrace)
        plotTopEvents(ofTrace, ofTracking, pIdx, axs=[topTenEventsAx, topTenTracesAx])
        topTenEventsAx.vlines(-10, -10, -5, lw=mpl.rcParams['axes.linewidth'])
        topTenEventsAx.hlines(-10, -10, -5, lw=mpl.rcParams['axes.linewidth'])
        topTenTracesAx.vlines(-12.5, -5, 5, lw=mpl.rcParams['axes.linewidth'])
        topTenTracesAx.hlines(-5, -12.5, -7.5, lw=mpl.rcParams['axes.linewidth'])
        
        # map
        mapAx = layout.axes[('n{}'.format(n),'ofMap')]['axis']
        fig.sca(mapAx)
        fv = fancyViz.TrackingIntensityPlot(session=ofSess, smoothing=15, saturation=1.0,
                                            portsUp=False, drawBg=False)
        fv.draw(ofTrace, ax=mapAx)
        headCoords = fv.coordinates
        mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1], marker='x',
                      linewidth=mpl.rcParams['axes.linewidth'],
                      c=cmocean.cm.phase(np.arange(10)/11))
        
        # 2-choice
        schemAx = layout.axes[('n{}'.format(n),'chSchematic')]['axis']
        fig.sca(schemAx)
        fv = fancyViz.SchematicIntensityPlot(chSess, linewidth=mpl.rcParams['axes.linewidth'],
                                             smoothing=5, splitReturns=False)
        img = fv.draw(chTrace, ax=schemAx)
        
        topTenEventsAx = layout.axes[('n{}'.format(n),'chTopTen')]['axis']
        topTenTracesAx = layout.axes[('n{}'.format(n),'chTopTenTraces')]['axis']
        pIdx = getTop10Events(chTrace)
        plotTopEvents(chTrace, chTracking, pIdx, axs=[topTenEventsAx, topTenTracesAx])
        topTenEventsAx.vlines(-10, -10, -5, lw=mpl.rcParams['axes.linewidth'])
        topTenEventsAx.hlines(-10, -10, -5, lw=mpl.rcParams['axes.linewidth'])
        topTenTracesAx.vlines(-12.5, -5, 5, lw=mpl.rcParams['axes.linewidth'])
        topTenTracesAx.hlines(-2, -12.5, -7.5, lw=mpl.rcParams['axes.linewidth'])
        
        mapAx = layout.axes[('n{}'.format(n),'chMap')]['axis']
        fig.sca(mapAx)
        fv = fancyViz.TrackingIntensityPlot(session=chSess, smoothing=15, saturation=1.0,
                                            portsUp=True, drawBg=False)
        fv.draw(chTrace, ax=mapAx)
        headCoords = fv.coordinates
        mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1],
                      marker='x', linewidth=mpl.rcParams['axes.linewidth'],
                      c=cmocean.cm.phase(np.arange(10)/11),
                      transform=fv.transform)
        
        plt.suptitle('{} {} {} #{}'.format(ofSess.meta.genotype, ofSess.meta.animal,
                                           ofSess.meta.date, neuron))
        
        layout.insert_figures('plots')
        #layout.write_svg(outputFolder / 'oftChoiceComp' / (svgName[:-4]+str(neuron)+'.svg'))
        pdf.savefig(fig)
        plt.close('all')
        
    #layout.insert_figures('plots')
    #layout.write_svg(outputFolder / svgName)
    pdf.close()


#%%######################################################
    def plotTopEvents(trace, tracking, pIdx, axs=None, framesBefore=5, framesAfter=15, offset=20):
    trace = trace.copy()
    tracking = tracking.copy()
    
    if not axs:
        fig, axs = plt.subplots(2, 1, figsize=(2.5, 1.25))
    ax, tax = axs
    
    tlen = framesBefore+framesAfter+1
    #traceOffset = 1.1 * tlen
    
    ts = []
    for i,idx in enumerate(pIdx):
        trackAll = tracking.loc[idx-framesBefore:idx+framesAfter].copy()
        body = trackAll['body'][['x','y']]
        origin = body.iloc[0]
        body -= origin
        head = (trackAll['leftEar'][['x','y']] + trackAll['rightEar'][['x','y']]) / 2
        head -= origin
        tail = trackAll['tailBase'][['x','y']]
        tail -= origin
        
        angle = np.arctan2(tail.iloc[0]['x'], tail.iloc[0]['y']) + np.deg2rad(180)
        body.loc[:,['x','y']] = rotate(body, angle=angle)
        head.loc[:,['x','y']] = rotate(head, angle=angle)
        tail.loc[:,['x','y']] = rotate(tail, angle=angle)
        
        color = cmocean.cm.phase(i/11)
        
        t = trace.loc[idx-framesBefore:idx+framesAfter].values
        #x = np.arange(i*traceOffset, i*traceOffset+tlen) - tlen/2
        tax.plot(t, color=color, clip_on=False, alpha=.75, lw=.5)
        
        y = i//5 * offset
        x = i%5
        ax.hlines([-y], x*offset-offset*.32, x*offset+offset*.32,
                  zorder=-99, alpha=.5, ls='-', lw=mpl.rcParams['axes.linewidth'])
        ax.vlines([x*offset], -y-offset*.32, -y+offset*.32, 
                  zorder=-99, alpha=.5, ls='-', lw=mpl.rcParams['axes.linewidth'])
        for n,f in enumerate(range(-framesBefore,framesAfter+1)):
            ax.plot(np.stack([head.loc[idx+f,'x']+x*offset,
                              body.loc[idx+f,'x']+x*offset, 
                              tail.loc[idx+f,'x']+x*offset]),
                    np.stack([head.loc[idx+f,'y']-y,
                              body.loc[idx+f,'y']-y,
                              tail.loc[idx+f,'y']-y]),
                    color=color, alpha=np.clip(((tlen-n)/tlen)**.8,0,1),
                    zorder=-y, clip_on=False)
#            ax.scatter(head.loc[idx+n,'x']+i*offset, head.loc[idx+n,'y'],
#                       marker='x', s=5, color=color, alpha=1-((n+5)/30)**.5, clip_on=False)
    
    ax.set_aspect('equal')
    ax.set_xlim(-offset, 4*offset+offset)
    ax.set_ylim(-.6*offset-offset, .6*offset)
    #tax.set_xlim([-traceOffset, i*traceOffset+traceOffset])
    #tax.set_ylim((0,12))
    ax.axis('off')
    tax.axis('off')
    
#%%
selTask = 'oft'
ofSess =  next(readSessions.findSessions(endoDataPath, task='openField',
                                         filterQuery='date != "190224"',
                                         animal='5308', date='190201'))
chSess = next(readSessions.findSessions(endoDataPath, task='2choice',
                                        genotype=ofSess.meta.genotype,
                                        animal=ofSess.meta.animal,
                                        date=ofSess.meta.date))

ofTraces = ofSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
chTraces = chSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
#ofTracking = ofSess.readTracking(inCm=True).reset_index(drop=True)
ofTracking = getSmoothedTracking(endoDataPath, ofSess.meta.genotype,
                                 ofSess.meta.animal, ofSess.meta.date,
                                 'openField')
#chTracking = chSess.readTracking(inCm=True).reset_index(drop=True)
chTracking = getSmoothedTracking(endoDataPath, ofSess.meta.genotype,
                                 ofSess.meta.animal, ofSess.meta.date,
                                 '2choice')

svgName = 'oftChoiceCompSmall2.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()
for n,neuron in enumerate([86,192]):
    ofTrace = ofTraces[neuron]
    chTrace = chTraces[neuron]

    # open field
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
    pIdx = getTop10Events(ofTrace)
    plotTopEvents(ofTrace, ofTracking, pIdx, axs=[topTenEventsAx, topTenTracesAxOf],
                  offset=10)
    topTenEventsAx.vlines(-7, -15, -10, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenEventsAx.hlines(-15, -7, -2, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxOf.vlines(-2.5, -5, 5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxOf.hlines(-5, -2.5, 2.5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    
    # map
    mapAx = layout.axes[('n{}'.format(n),'ofMap')]['axis']
    fig.sca(mapAx)
    fv = fancyViz.TrackingIntensityPlot(session=ofSess, smoothing=20, saturation=1,
                                        portsUp=False, drawBg=False)
    fv.draw(ofTrace, ax=mapAx)
    headCoords = pd.DataFrame(fv.coordinates).fillna(method='ffill', axis=0).values
    mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1], marker='x',
                  linewidth=mpl.rcParams['axes.linewidth'],
                  c=cmocean.cm.phase(np.arange(10)/11))
    
    # 2-choice
    schemAx = layout.axes[('n{}'.format(n),'chSchematic')]['axis']
    fig.sca(schemAx)
    fv = fancyViz.SchematicIntensityPlot(chSess, linewidth=mpl.rcParams['axes.linewidth'],
                                         smoothing=5, splitReturns=False, saturation=1)
    img = fv.draw(chTrace, ax=schemAx)
    
    topTenEventsAx = layout.axes[('n{}'.format(n),'chTopTen')]['axis']
    topTenTracesAxCh = layout.axes[('n{}'.format(n),'chTopTenTraces')]['axis']
    pIdx = getTop10Events(chTrace)
    plotTopEvents(chTrace, chTracking, pIdx, axs=[topTenEventsAx, topTenTracesAxCh],
                  offset=10)
    topTenEventsAx.vlines(-7, -15, -10, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenEventsAx.hlines(-15, -7, -2, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxCh.vlines(-2.5, -5, 5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxCh.hlines(-5, -2.5, 2.5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    
    mapAx = layout.axes[('n{}'.format(n),'chMap')]['axis']
    fig.sca(mapAx)
    fv = fancyViz.TrackingIntensityPlot(session=chSess, smoothing=20, saturation=1.0,
                                        portsUp=True, drawBg=False)
    fv.draw(chTrace, ax=mapAx)
    headCoords = pd.DataFrame(fv.coordinates).fillna(method='ffill', axis=0).values
    mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1],
                  marker='x', linewidth=mpl.rcParams['axes.linewidth'],
                  c=cmocean.cm.phase(np.arange(10)/11),
                  transform=fv.transform)
    
    traceAxLims = np.concatenate([topTenTracesAxCh.get_ylim(),topTenTracesAxOf.get_ylim()])
    traceAxLims = (traceAxLims.min(), traceAxLims.max())
    topTenTracesAxCh.set_ylim(traceAxLims)
    topTenTracesAxOf.set_ylim(traceAxLims)
    
    #plt.suptitle('{} {} {} #{}'.format(ofSess.meta.genotype, ofSess.meta.animal,
    #                                   ofSess.meta.date, neuron))
    
layout.insert_figures('plots')
layout.write_svg(outputFolder / 'oftChoiceComp' / (svgName[:-4]+str(neuron)+'.svg'))

plt.show()


#%%####################################################
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()


#%%
svgName = 'trajectory.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()

##%%
chSess = next(readSessions.findSessions('data/endoData_2019.hdf', genotype='oprm1',
                                        animal='5703', date='190130'))
chTracking = getSmoothedTracking(endoDataPath, 'oprm1','5703','190130','2choice')

##%%
ax0, ax1, ax2 = [layout.axes['{}TrajMap'.format(tt)]['axis'] for tt in ['rst','ost','osw']]

chTracking['bin'] = chTracking.actionProgress * 100 // 10

rst = chTracking.loc[chTracking.behavior.str.startswith('mR2Cr.')]
rst_mean = rst.groupby('bin').mean()
for actionNo, track in rst.groupby('actionNo'):
    plot2CTrackingEvent(track, ax0, color=style.getColor('r.'), alpha=.025)
plot2CTrackingEvent(rst_mean, ax0, color='k', lw=.5, alpha=.5)

ost = chTracking.loc[chTracking.behavior.str.startswith('mR2Co.')]
ost_mean = ost.groupby('bin').mean()
for actionNo, track in ost.groupby('actionNo'):
    plot2CTrackingEvent(track, ax1, color=style.getColor('o.'), alpha=.025)
plot2CTrackingEvent(ost_mean, ax1, color='k', lw=.5, alpha=.5)

osw = chTracking.loc[chTracking.behavior.str.startswith('mR2Co!')]
osw_mean = osw.groupby('bin').mean()
for actionNo, track in osw.groupby('actionNo'):
    plot2CTrackingEvent(track, ax2, color=style.getColor('o!'), alpha=.025)
plot2CTrackingEvent(osw_mean, ax2, color='k', lw=.5, alpha=.5)

#plot2CTrackingEvent(rst_mean, ax3, color=style.getColor('r.'), alpha=.5, lw=.5)
#plot2CTrackingEvent(ost_mean, ax3, color=style.getColor('o.'), alpha=.5, lw=.5)
#plot2CTrackingEvent(osw_mean, ax3, color=style.getColor('o!'), alpha=.5, lw=.5)

for ax in [ax0, ax1,ax2]:
    t = ax.transData
    t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(15/2, 15/2, 90) + t
    corners_x, corners_y = [0,0,15,15,0], [0,15,15,0,0]
    ax.plot(corners_x, corners_y, 'k', lw=0.5, transform=t)
    s = 15/7
    for y in s*np.array([1, 3, 5]):
        drawRoundedRect(ax, (15, y), s, s, [0, 0, s/4, s/4],
                        fill=False, edgecolor="k", lw=mpl.rcParams['axes.linewidth'],
                        transform=t)

ax2.set_xlabel('right to center turn\nmovement trajectories', labelpad=3)
ax0.set_ylabel('win-stay', color=style.getColor('r.'))
ax1.set_ylabel('lose-stay', color=style.getColor('o.'))
ax2.set_ylabel('lose-switch', color=style.getColor('o!'))
for ax in [ax0,ax1,ax2]:
    ax.set_aspect('equal')
    ax.set_xlim((-1, 16))
    ax.set_ylim((7.5, 18))
    ax.set_xticks(())
    ax.set_yticks(())
    ax.yaxis.set_label_coords(0, .35)
    sns.despine(top=True, left=True, right=True, bottom=True, ax=ax)


###%%
#trajectories = chTracking.loc[chTracking.behavior.isin(['mR2Cr.','mR2Co.','mR2Co!'])]
#action2behavior = trajectories.groupby('actionNo').behavior.first().str.slice(-2)
#trajectories = (trajectories.groupby(['actionNo','bin']).mean()
#                            [['body','tailBase','leftEar','rightEar']])
#pairIdxs = itertools.combinations(trajectories.index.get_level_values(0).unique(), 2)
#out = []
#for idx in pairIdxs:
#    dist = get2CTrajectoryDist(trajectories.loc[idx[0]], trajectories.loc[idx[1]])
#    tts = '{}X{}'.format(*sorted([action2behavior.loc[idx[0]], action2behavior.loc[idx[1]]]))
#    out.append([tts, dist])
#df = pd.DataFrame(out, columns=['trialTypes','distance'])   
#df.to_pickle('cache/oprm1_5703_mR2C_trajectories.pkl')

df = pd.read_pickle('cache/oprm1_5703_mR2C_trajectories.pkl')

##%%
axs = [layout.axes['{}TrajKde'.format(tt)]['axis'] for tt in ['rst',]]

pdict = {'rst':(('r.Xr.','o.Xr.','o!Xr.'),
                (style.getColor('r.'),style.getColor('o.'), style.getColor('o!'))),
         'ost':(('o.Xo.','o.Xr.','o!Xo.'),
                (style.getColor('o.'),style.getColor('r.'), style.getColor('o!'))),
         'osw':(('o!Xo!','o!Xr.','o!Xo.'),
                (style.getColor('o!'),style.getColor('r.'), style.getColor('o.')))}

for p,tt in enumerate(('rst',)):
    ax = axs[p]
    strs, cs = pdict[tt]
    
    sns.distplot(df.loc[df.trialTypes == strs[0],'distance'], hist=False, color=cs[0],
                 kde_kws={'clip_on':True, 'alpha':.75, 'cut':3}, ax=ax)
    sns.distplot(df.loc[df.trialTypes == strs[1],'distance'], hist=False, color=cs[1],
                 kde_kws={'clip_on':True, 'alpha':.75, 'cut':3}, ax=ax)
    sns.distplot(df.loc[df.trialTypes == strs[2],'distance'], hist=False, color=cs[2],
                 kde_kws={'clip_on':True, 'alpha':.75, 'cut':3}, ax=ax)
    
    ax.set_ylim((0,2))
    ax.set_yticks((0,1,2))
    ax.set_yticks((.5,1.5), minor=True)
    ax.set_yticklabels(())
    ax.set_ylabel('')
    ax.set_xlim((0,5))
    ax.set_xticks(np.arange(6))
    ax.set_xticks(np.arange(5)+.5, minor=True)
    ax.set_xticklabels(())
    ax.set_xlabel('')
    #if tt == 'osw':
    ax.set_xticklabels(ax.get_xticks())
    ax.set_xlabel('pairwise trajectory\ndistance (cm)')
    #if tt == 'ost':
    ax.set_yticklabels(ax.get_yticks())
    ax.set_ylabel('density')
    sns.despine(ax=ax, offset=.1)

legend_elements = [mpl.patches.Patch(color=color,
                                     label={'r.Xr.':'win-stay',
                                            'o.Xr.':'lose-stay',
                                            'o!Xr.':'lose-switch'}[tt]) 
                       for tt, color in zip(*pdict['rst'])]
lg = axs[0].legend(handles=legend_elements, ncol=1, title='win-stay vs ...',
                   bbox_to_anchor=(.96,.96), loc='upper right')
lg.get_title().set_fontsize(6)


##%%
aDurations = chTracking.loc[chTracking.behavior.isin(['mR2Cr.','mR2Co.','mR2Co!'])]
aDurations = aDurations.groupby(['behavior','actionNo']).size() / 20.

ax = layout.axes['durationKde']['axis']
for action, ds in aDurations.groupby('behavior'):
    sns.distplot(ds, hist=False, color=style.getColor(action[-2:]),
                 kde_kws={'alpha':.75}, ax=ax)

ax.set_xlim((0,2))
ax.set_xticks(np.arange(0,2.2,.5))
ax.set_xticks(np.arange(0,2,.25), minor=True)
ax.set_xlabel('turn duration (s)')
ax.set_ylim((0,4))
ax.set_yticks((0,2,4))
ax.set_yticks((1,3), minor=True)
ax.set_ylabel('density')
sns.despine(ax=ax)


##%%
exampleNeurons = [100,26]
s = next(readSessions.findSessions(endoDataPath, genotype='oprm1',
                                   animal='5703', date='190130', task='2choice'))
traces = s.readDeconvolvedTraces(rScore=True)
lfa = s.labelFrameActions(reward='fullTrial', switch=True).set_index(traces.index)
for p,n in enumerate(exampleNeurons):
    trace = traces[n]  
    for trialType in ['r.','o.','o!']:
        axfv = layout.axes['f8_ex{}_{}'.format(p+1, trialType)]['axis']
        fv = fancyViz.SchematicIntensityPlot(s, splitReturns=False,
                                             linewidth=mpl.rcParams['axes.linewidth'],
                                             smoothing=7.5, saturation=2)
        fv.setMask(lfa.label.str.endswith(trialType).values)
        img = fv.draw(trace, ax=axfv)
        axfv.axis('on')
        axfv.set_xlabel({'r.':'win-stay','o.':'lose-stay','o!':'lose-switch'}[trialType],
                        color=style.getColor(trialType))
        axfv.set_xticks(()); axfv.set_yticks(())
        sns.despine(ax=axfv, left=True, bottom=True)

cbs = []
for c in [1,2]:
    cax = layout.axes['colorbar'+str(c)]['axis']
    cb = plt.colorbar(img, cax=cax, orientation='horizontal')
    cax.text(-2.05, -.3, '-2', ha='right', va='center', fontdict={'fontsize':6})
    cax.text(2.05, -.3, '2', ha='left', va='center', fontdict={'fontsize':6})
    cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})
    cax.axis('off')
    cbs.append(cb)
[cb.outline.set_visible(False) for cb in cbs]


##%%
def getActionMeans(endoDataPath, genotype, animal, date):
    s = next(readSessions.findSessions(endoDataPath, task='2choice',
                                       genotype=genotype, animal=animal, date=date))
    lfa = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
    deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)

    if not len(lfa) == len(deconv):
        print(str(s)+': more labeled frames than signal!')
        return -1

    means = pd.DataFrame(deconv.groupby([lfa['label'], lfa['actionNo']]).mean().stack(),
                         columns=['trialMean'])
    means.index.names = means.index.names[:2] + ['neuron']
    means.reset_index(inplace=True)
    means['action'] = means.label.str.slice(0,4)

    return(means)

##%%
def getActionWindows(endoDataPath, genotype, animal, date,
                     win_size=(20, 19), nan_actions=True,
                     incl_actions=["pL2C","pR2C","mL2C","mR2C","pC2L","pC2R",
                                   "mC2L","mC2R","dL2C","dR2C"],
                     incl_trialTypes=["r.","o.","o!"]):
    s = next(readSessions.findSessions(endoDataPath, task='2choice',
                                       genotype=genotype, animal=animal, date=date))
    lfa = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
    deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    
    if not len(lfa) == len(deconv):
        print(str(s)+': more labeled frames than signal!')
        return -1

    lfa['index'] = deconv.index
    deconv = deconv.set_index([lfa.label,lfa.actionNo], append=True)
    deconv.index.names = ['index','label','actionNo']
    deconv.columns.name = 'neuron'
    
    lfa = lfa.loc[lfa.label.str.slice(0,4).isin(incl_actions) &
                  lfa.label.str.slice(4).isin(incl_trialTypes)]
    actions_idx = lfa.groupby('actionNo')[['index','actionNo','label']].first().values
    
    windows = []
    neurons = deconv.columns
    for idx, actionNo, label in actions_idx:
        win = deconv.loc[idx-win_size[0]:idx+win_size[1]].reset_index()
        if nan_actions:
            win.loc[win.actionNo > actionNo, neurons] = np.nan
            win.loc[win.actionNo < actionNo-1, neurons] = np.nan
        win['frameNo'] = np.arange(len(win))
        win['label'] = label
        win['actionNo'] = actionNo
        win = win.set_index(['actionNo','label','frameNo'])[neurons]
        win = win.unstack('frameNo').stack('neuron')
        win.columns = pd.MultiIndex.from_product([['frameNo'], win.columns])
        windows.append(win.reset_index())
    windows = pd.concat(windows, ignore_index=True)
    windows['action'] = windows.label.str.slice(0,4)
    windows['trialType'] = windows.label.str.slice(4)
    return windows

##%%
def getPhaseLabels(phase):
    if 'S' in phase:
        actions = [phase.replace('S', s) for s in 'LR']
        inclLabels = [actions[0]+tt for tt in ['r.','o.','o!']] + [actions[1]+tt for tt in ['o!','o.','r.']]
    else:
        inclLabels = [phase+tt for tt in ['r.','o.','o!']]
    return inclLabels


def avRegPlot(means, phase='mS2C', ax=None):
    inclLabels = getPhaseLabels(phase)
    data = means.loc[means.label.isin(inclLabels)]
    for l, ldata in data.groupby('label'):
        ax.scatter(ldata['value'], ldata['trialMean'],
                   facecolor='w', edgecolor=style.getColor(l[-2:]),
                   #marker='<' if 'L' in l else '>',
                   marker='o', alpha=.25, s=3.5, lw=.5, clip_on=True)
        ax.errorbar(ldata['value'].mean(), ldata['trialMean'].mean(),
                    xerr=ldata['value'].sem(), yerr=ldata['trialMean'].sem(),
                    color=sns.desaturate(style.getColor(l[-2:]),.8),
                    #marker='<' if 'L' in l else '>',
                    marker='o', ms=3, clip_on=False)
    sns.regplot('value', 'trialMean', data=data, fit_reg=True, scatter=False,
                ax=ax, color='k', ci=False, line_kws={'zorder':-99, 'lw':.5})
    
    
def avAvgTracePlot(wins, phase='mS2C', compression=40, ax=None):
    inclLabels = getPhaseLabels(phase)
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for l,ldata in wins.loc[wins.label.isin(inclLabels)].groupby('label'):
        x = np.array(ldata['frameNo'].columns.values / compression + ldata['value'].mean(),
                     dtype='float')
        x_offset = -(len(x) // 2) / compression
        y = ldata['frameNo'].mean().values
        y[ldata['frameNo'].notna().sum(axis=0) < 20] = np.nan
        sem = ldata['frameNo'].sem().values
        sem[ldata['frameNo'].notna().sum(axis=0) < 20] = np.nan
        ax.fill_between(x + x_offset, y-sem, y+sem, clip_on=False,
                        color=style.getColor(l[-2:]), lw=0, alpha=.5)
        ax.plot(x + x_offset, y, color=style.getColor(l[-2:]), alpha=.8,
                clip_on=False)
        ax.axvline(ldata['value'].mean(), ls=':', color='k', alpha=1,
                   lw=mpl.rcParams['axes.linewidth'])


##%%
#means = getActionMeans(endoDataPath, 'oprm1', '5703', '190130')
#wins = getActionWindows(endoDataPath, 'oprm1', '5703', '190130')
##%%
means = pd.read_pickle('cache/oprm1_5703_190130_actionMeans.pkl')
means = means.loc[means.neuron.isin(exampleNeurons)].set_index('actionNo').sort_index()
wins = pd.read_pickle('cache/oprm1_5703_190130_actionWindows.pkl')
wins = wins.loc[wins.neuron.isin(exampleNeurons)].set_index('actionNo').sort_index()
#wins['frameNo']  = wins['frameNo'].rolling(3, center=True, axis=1).mean()
actionValues = pd.read_pickle('cache/actionValues.pkl')
actionValues.set_index(['genotype','animal','date','actionNo'], inplace=True)
actionValues.sort_index(inplace=True)

av = actionValues.loc[('oprm1', '5703', '190130')].copy()
means['value'] = av.value
wins['value'] = av.value
means = means.reset_index().set_index(['neuron','actionNo']).sort_index()
wins = wins.reset_index().set_index(['neuron','actionNo']).sort_index()


##%%
for p, neuron in enumerate(exampleNeurons):
    regAx = layout.axes['ac1_ex{}'.format(p+1)]
    avgAx = layout.axes['ac2_ex{}'.format(p+1)]
    avRegPlot(means.loc[neuron],phase='mR2C',ax=regAx)
    avAvgTracePlot(wins.loc[neuron],phase='mR2C',compression=15,ax=avgAx)
    
    for ax in [regAx, avgAx]:
        ax.set_ylim((-.75,6))
        ax.set_yticks((0,2,4,6))
        ax.set_yticks((1,3,5), minor=True)
        ax.set_xlim((-1,5))
        ax.set_ylabel('z-score')
    avgAx.set_xticks(np.arange(-1,6))
    avgAx.set_xticks(np.arange(-1,5,.5), minor=True)
    regAx.set_xticks(())
    avgAx.set_xlabel('action value')
    regAx.set_xlabel('')
    sns.despine(ax=avgAx, trim=False)
    sns.despine(ax=regAx, bottom=True, trim=True)
    
##%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)
plt.close('all')


#%%
tuningData = (pd.concat([chTuningData, ofTuningData], keys=['choice','oft'], names=['task'])
                .reset_index('task').reset_index(drop=True))
maxTunings = (tuningData.loc[tuningData.groupby(['task','genotype','animal','date','neuron'])
                                        .tuning.apply(lambda ts: ts.abs().idxmax()).values])
maxTunings = maxTunings.set_index(['task','genotype','animal','date','neuron']).sort_index()
maxTunings = maxTunings[['tuning','pct']].unstack('task').dropna()

plt.scatter(maxTunings.tuning.choice, maxTunings.tuning.oft,
            c=maxTunings.reset_index().genotype.apply(style.getColor).values)
plt.gca().set_aspect('equal')

#%%
tuningData = (pd.concat([chTuningData, ofTuningData], keys=['choice','oft'], names=['task'])
                .reset_index('task').reset_index(drop=True))
tuningData['signp'] = tuningData.pct > .995
tuningData = tuningData.set_index(['task','action','genotype','animal','date','neuron'])

data = tuningData.unstack(['task','action']).dropna()['signp']

data[('choice','none')] = data.choice.sum(axis=1) == 0
data[('choice','leftTurn')] = data.choice[['mC2L-','mR2C-']].any(axis=1)
data[('choice','rightTurn')] = data.choice[['mC2R-','mC2R-']].any(axis=1)
data[('choice','port')] = data.choice.loc[:,data.choice.columns.str.contains('^p|^d')].any(axis=1)
#%%
data = data.loc[:,data.columns.get_level_values(1).isin(['leftTurn'])] #,'rightTurn','running',
                                                         #'stationary','port','none'])]
    
#%%
noNeurons = data.groupby(['genotype','animal','date']).size()
pctTund = data.groupby(['genotype','animal','date']).sum(axis=1) / noNeurons.values[:,np.newaxis]
pctTund *= 100

obsOvlp = ((data.oft * data.choice).groupby(['genotype','animal','date']).sum() / \
            noNeurons.values[:,np.newaxis])
expOvlp = ((data.oft.groupby(['genotype','animal','date']).sum() / \
            noNeurons.values[:,np.newaxis]) * \
           (data.choice.groupby(['genotype','animal','date']).sum() / \
            noNeurons.values[:,np.newaxis]))
ovlpDf = pd.concat([obsOvlp, expOvlp], axis=1, keys=['observed','expected']) * 100

fig, axs = plt.subplots(2,1, sharex=True, sharey=True, figsize=(.8,2.3),
                        gridspec_kw={'hspace':.35})
for gt,df in pctTund.groupby('genotype'):
    x = {'d1':0,'a2a':1,'oprm1':2}[gt]
    ax = axs[0]
    ax.errorbar(x=x-.2, y=df.oft.mean(), yerr=df.oft.sem(),
                marker='^', color=style.getColor(gt), clip_on=False)
    ax.errorbar(x=x+.2, y=df.choice.mean(), yerr=df.choice.sem(),
                 marker='s', color=style.getColor(gt), clip_on=False)
    #ax.plot([x-.2,x+.2],[df.oft.mean(), df.choice.mean()],
    #        color=style.getColor(gt), zorder=-99)
    ax.plot([[x-.2]*len(df),[x+.2]*len(df)], df.T.values,
            color=style.getColor(gt), alpha=.25, zorder=-99, clip_on=False)
    ax.axhline(0, ls=':', alpha=.5, color='k', zorder=-100,
               lw=mpl.rcParams['axes.linewidth'])

legend_elements = [mpl.lines.Line2D([0], [0], marker='^', ls='', color='k',
                                    label='open field'),
                   mpl.lines.Line2D([0], [0], marker='s', ls='', color='k',
                                    label='2-choice')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(.5,1.1))

ax.set_ylabel('% left turn-tuned')
sns.despine(ax=ax)

for gt,df in ovlpDf.groupby('genotype'):
    x = {'d1':0,'a2a':1,'oprm1':2}[gt]
    ax = axs[1]
    ax.errorbar(x=x-.2, y=df.observed.mean(), yerr=df.observed.sem(),
                marker='o', color=style.getColor(gt), clip_on=False)
    ax.errorbar(x=x+.2, y=df.expected.mean(), yerr=df.observed.sem(),
                marker='o', markeredgecolor=style.getColor(gt),
                markerfacecolor='w', ecolor=style.getColor(gt), clip_on=False)
    #ax.plot([x-.2,x+.2],[df.observed.mean(), df.expected.mean()],
    #        color=style.getColor(gt), zorder=-99)
    ax.plot([[x-.2]*len(df),[x+.2]*len(df)], df.T.values,
            color=style.getColor(gt), alpha=.25, zorder=-99, clip_on=False)
    ax.axhline(0, ls=':', alpha=.5, color='k', zorder=-100,
               lw=mpl.rcParams['axes.linewidth'])
    
legend_elements = [mpl.lines.Line2D([0], [0], marker='o', ls='', color='k',
                                    label='observed'),
                   mpl.lines.Line2D([0], [0], marker='o', ls='', markeredgecolor='k',
                                    markerfacecolor='w', label='expected')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(.5,1.1))

ax.set_ylim(-10,100)
ax.set_yticks([25,75], minor=True)
ax.set_xlim((-.5,2.5))
ax.set_xticks([0,1,2])
ax.set_xticklabels(['D1','A2A','Oprm1'], rotation=25, ha='right')
ax.set_ylabel('% overlap')
sns.despine(ax=ax)

fig.savefig('svg/leftTurnOverlap.svg', bbox_inches='tight', pad_inches=0)   

 
#%%
plt.figure()
for gt, df in data.groupby('genotype'):
    exp = df.choice[['none','leftTurn','rightTurn','port']].sum(axis=0) / len(df)
    obs = df.loc[df['oft','leftTurn']].choice.sum(axis=0) / len(df.loc[df['oft','leftTurn']])
    #plt.scatter(exp, obs, color=[style.getColor(a[:-1]) for a in exp.index],
    #            s=100, marker='s', zorder=-99)
    plt.scatter(exp.loc[~exp.index.isin(['leftTurn'])],
                obs.loc[~exp.index.isin(['leftTurn'])],
                edgecolor=style.getColor(gt), marker='o', s=50,
                lw=1, facecolor='none')
    plt.scatter(exp.loc[exp.index.isin(['leftTurn'])],
                obs.loc[exp.index.isin(['leftTurn'])],
                color=style.getColor(gt), marker='o', s=50, lw=1)
    plt.plot([0,.75], [0,.75], 'k--', alpha=.5, zorder=-99)
    plt.gca().set_aspect('equal')
    sns.despine()
        
#%%
