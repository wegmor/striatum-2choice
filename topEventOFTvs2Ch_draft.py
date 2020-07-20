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
from utils import readSessions, fancyViz
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

def plotTop10Events(trace, tracking, axs=None, framesBefore=5, framesAfter=15, offset=20):
    trace = trace.copy()
    tracking = tracking.copy()
    
    if not axs:
        fig, axs = plt.subplots(2, 1, figsize=(2.5, 1.25))
    ax, tax = axs
        
    # max events at least 1 sec separated from other included events
    pIdx, pProp = find_peaks(trace, distance=max(framesBefore,framesAfter), width=1)
    # find top 10 most prominent (not necessarily highest!) events
    sortIdx = np.argsort(pProp['prominences'])
    pIdx = pIdx[sortIdx]
    pIdx = pIdx[(pIdx >= framesBefore) & (pIdx <= pIdx.max() - framesAfter)][-10:]
    
    tlen = framesBefore+framesAfter+1
    traceOffset = 1.25 * tlen

    for i,idx in enumerate(pIdx):
        trackAll = tracking.loc[idx-framesBefore:idx+framesAfter]
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
        for n in range(-5,16):
            ax.plot(np.stack([head.loc[idx+n,'x']+i*offset,
                              body.loc[idx+n,'x']+i*offset, 
                              tail.loc[idx+n,'x']+i*offset]),
                    np.stack([head.loc[idx+n,'y'], body.loc[idx+n,'y'], tail.loc[idx+n,'y']]),
                    color=color, alpha=1-((n+5)/30)**.5, zorder=-99, clip_on=False)
#            ax.scatter(head.loc[idx+n,'x']+i*offset, head.loc[idx+n,'y'],
#                       marker='x', s=5, color=color, alpha=1-((n+5)/30)**.5, clip_on=False)
    
    ax.set_aspect('equal')
    ax.set_xlim(-offset, i*offset+offset)
    ax.set_ylim(-.6*offset, .6*offset)
    tax.set_xlim([-traceOffset, i*traceOffset+traceOffset])
    ax.axis('off')
    tax.axis('off')

    return np.array(pIdx)


#%%
ofTuningData = analysisOpenField.getTuningData(endoDataPath)
chTuningData = analysisTunings.getTuningData(endoDataPath)

    
#%%
#svgName = 'oftChoiceComp.svg'
#layout = figurefirst.FigureLayout(templateFolder / svgName)
#layout.make_mplfigures()

for ofSess in readSessions.findSessions(endoDataPath, task='openField',
                                        filterQuery='date != "190224"'):
    pdf = PdfPages('svg/oftChoiceComp/{}_{}_{}.pdf'.format(ofSess.meta.genotype,
                                                           ofSess.meta.animal,
                                                           ofSess.meta.date))
    
    chSess = next(readSessions.findSessions(endoDataPath, task='2choice',
                                            genotype=ofSess.meta.genotype,
                                            animal=ofSess.meta.animal,
                                            date=ofSess.meta.date))
    
    ofTraces = ofSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    chTraces = chSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    ofTracking = ofSess.readTracking(inCm=True).reset_index(drop=True)
    chTracking = chSess.readTracking(inCm=True).reset_index(drop=True)
    
    ofTuning = ofTuningData.query('animal == @ofSess.meta.animal & date == @ofSess.meta.date')
    chTuning = chTuningData.query('animal == @ofSess.meta.animal & date == @ofSess.meta.date')
    tuning = pd.concat([chTuning, ofTuning], keys=['choice','oft'])
    neurons = (tuning.groupby('neuron').tuning.max()
                     .sort_values(ascending=False).index.astype('int').values)
    
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
                                             smoothing=3)
        img = fv.draw(ofTrace, ax=schemAx)
        
        # top events
        topTenEventsAx = layout.axes[('n{}'.format(n),'ofTopTen')]['axis']
        topTenTracesAx = layout.axes[('n{}'.format(n),'ofTopTenTraces')]['axis']
        pIdx = plotTop10Events(ofTrace, ofTracking, axs=[topTenEventsAx, topTenTracesAx])
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
        pIdx = plotTop10Events(chTrace, chTracking, axs=[topTenEventsAx, topTenTracesAx])
        topTenEventsAx.vlines(-10, -10, -5, lw=mpl.rcParams['axes.linewidth'])
        topTenEventsAx.hlines(-10, -10, -5, lw=mpl.rcParams['axes.linewidth'])
        topTenTracesAx.vlines(-12.5, -2, 3, lw=mpl.rcParams['axes.linewidth'])
        topTenTracesAx.hlines(-2, -12.5, -7.5, lw=mpl.rcParams['axes.linewidth'])
        
        mapAx = layout.axes[('n{}'.format(n),'chMap')]['axis']
        fig.sca(mapAx)
        fv = fancyViz.TrackingIntensityPlot(session=chSess, smoothing=15, saturation=1.0,
                                            portsUp=True, drawBg=False)
        fv.draw(chTrace, ax=mapAx)
        headCoords = fv.coordinates
        mapAx.scatter(headCoords[pIdx,1], fv.canvasSize[1]-headCoords[pIdx,0],
                      marker='x', linewidth=mpl.rcParams['axes.linewidth'],
                      c=cmocean.cm.phase(np.arange(10)/11))
        
        plt.suptitle('{} {} {} #{}'.format(ofSess.meta.genotype, ofSess.meta.animal,
                                           ofSess.meta.date, neuron))
        
        layout.insert_figures('plots')
        #layout.write_svg(outputFolder / 'oftChoiceComp' / (svgName[:-4]+str(neuron)+'.svg'))
        pdf.savefig(fig)
        plt.close('all')
        
    #layout.insert_figures('plots')
    #layout.write_svg(outputFolder / svgName)
    pdf.close()


#%%
