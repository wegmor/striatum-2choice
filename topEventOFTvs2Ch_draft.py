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
chTracking['bin'] = chTracking.actionProgress * 100 // 10
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharex=True, sharey=True,
                                         gridspec_kw={'wspace':0.2})
[ax.set_aspect('equal') for ax in [ax0,ax1,ax2,ax3]]
#[ax.axis('off') for ax in [ax0,ax1,ax2,ax3]]
#ax0.set_xlim((0, 15))
#ax0.set_ylim((0, 15))
a=90

rst = chTracking.loc[chTracking.behavior.str.startswith('mL2Cr.')]
rst_mean = rst.groupby('bin').mean()
for actionNo, track in rst.groupby('actionNo'):
    plotTrackingEvent(track, ax0, rotateAngle=a, subtractOrigin=False,
                      color=style.getColor('r.'), alpha=.05)
plotTrackingEvent(rst_mean, ax0, rotateAngle=a, subtractOrigin=False,
                  color='k')

ost = chTracking.loc[chTracking.behavior.str.startswith('mL2Co.')]
ost_mean = ost.groupby('bin').mean()
for actionNo, track in ost.groupby('actionNo'):
    plotTrackingEvent(track, ax1, rotateAngle=a, subtractOrigin=False,
                      color=style.getColor('o.'), alpha=.05)
plotTrackingEvent(ost_mean, ax1, rotateAngle=a, subtractOrigin=False,
                  color='k')

osw = chTracking.loc[chTracking.behavior.str.startswith('mL2Co!')]
osw_mean = osw.groupby('bin').mean()
for actionNo, track in osw.groupby('actionNo'):
    plotTrackingEvent(track, ax2, rotateAngle=a, subtractOrigin=False,
                      color=style.getColor('o!'), alpha=.05)
plotTrackingEvent(osw_mean, ax2, rotateAngle=a, subtractOrigin=False,
                  color='k')

plotTrackingEvent(rst_mean, ax3, rotateAngle=a, subtractOrigin=False,
                  color=style.getColor('r.'))
plotTrackingEvent(ost_mean, ax3, rotateAngle=a, subtractOrigin=False,
                  color=style.getColor('o.'))
plotTrackingEvent(osw_mean, ax3, rotateAngle=a, subtractOrigin=False,
                  color=style.getColor('o!'))


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
