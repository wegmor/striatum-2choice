#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:19:30 2020

@author: mowe
"""

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
from utils import readSessions, segmentBehaviors, particleFilter
from utils.cachedDataFrame import cachedDataFrame


#%%
def getSmoothedTracking(dataFile, genotype, animal, date, task):
    def _medSmooth(tracking):
        tracking = tracking.copy()
        for col in ['leftEar','rightEar','body','tailBase']:
            _tracking = tracking[col].copy()
            _tracking.loc[_tracking.likelihood<.99,['x','y']] = np.nan
            _tracking.loc[:,['x','y']] = (_tracking[['x','y']].rolling(5, center=True, min_periods=3)
                                                              .median())
            tracking.loc[:,col] = _tracking.values
        return tracking
    
    @cachedDataFrame('smoothed_{}_tracking_{}-{}-{}.pkl'.format(task, genotype, animal, date))
    def _smoothTracking():
        sess = next(readSessions.findSessions(dataFile, task=task, genotype=genotype,
                                              animal=animal, date=date))
        tracking = sess.readTracking(inCm=True).reset_index(drop=True)
        tracking = _medSmooth(tracking)
        
        if task == 'openField':
            coords = particleFilter.particleFilter(tracking, flattening = 1e-12)
            coords.rename(columns={"bodyAngle": "bodyDirection"}, inplace=True)
            coords.rename_axis("time", axis=0, inplace=True)
            coords['bodyDirection'] = np.rad2deg(coords.bodyDirection)
            behaviors = segmentBehaviors.segmentBehaviors(coords)[['startFrame','behavior']]
            behaviors.insert(0, "actionNo", behaviors.index.copy())
            coords.reset_index(inplace=True)
            coords.rename(columns={'time':'frame'}, inplace=True)
            coords = coords.merge(behaviors, left_on='frame', right_on='startFrame',
                                  how='left').fillna(method='ffill')
            tracking['actionNo'] = coords.actionNo
            tracking['behavior'] = coords.behavior
        else:
            behaviors = sess.labelFrameActions(reward='fullTrial', switch=True).reset_index(drop=True)
            tracking['actionNo'] = behaviors.actionNo
            tracking['actionProgress'] = behaviors.actionProgress
            tracking['behavior'] = behaviors.label
        
        return tracking
    
    return _smoothTracking()


# https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python#34374437
def rotate(p, angle=0, origin=(0, 0)):
    #angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def plotTop10Events(trace, tracking, axs, framesBefore=5, framesAfter=14, offset=10):
    def _getTop10Events(trace, tracking, framesBefore, framesAfter):
        # make sure max event windows do not overlap
        pIdx, pProp = find_peaks(trace, distance=max(framesBefore,framesAfter), width=1)
        # find top 10 most prominent (not necessarily highest!) events
        sortIdx = np.argsort(pProp['prominences'])
        pIdx = pIdx[sortIdx]
        pIdx = pIdx[(pIdx >= framesBefore) & (pIdx <= pIdx.max() - framesAfter)]
        pIdx = pIdx[[~tracking.loc[idx-framesBefore:idx+framesAfter].isna().any(axis=1).any()
                        for idx in pIdx]]
        return np.array(pIdx)[:-11:-1]
    
    trace = trace.copy()
    tracking = tracking.copy()
    pIdx = _getTop10Events(trace, tracking, framesBefore, framesAfter)
    tlen = framesBefore + framesAfter + 1
    ax, tax = axs
    
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
    
    ax.set_aspect('equal')
    ax.set_xlim(-offset, 4*offset+offset)
    ax.set_ylim(-.6*offset-offset, .6*offset)
    ax.axis('off')
    tax.axis('off')
    
    return pIdx


def plot2CTrackingEvent(track, ax, color='k', lw=.5, alpha=1., portsUp=True):
    body = track['body'][['x','y']]
    head = (track['leftEar'][['x','y']] + track['rightEar'][['x','y']]) / 2
    tail = track['tailBase'][['x','y']]
    
    t = ax.transData
    if portsUp:
        t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(15/2, 15/2, 90) + t
    
    for n in range(len(head)):
        ax.plot(np.stack([head.iloc[n].x, body.iloc[n].x, tail.iloc[n].x]),
                np.stack([head.iloc[n].y, body.iloc[n].y, tail.iloc[n].y]),
                color=color, lw=lw, alpha=alpha,
                #alpha=np.clip(((len(head)-n)/len(head))**.8,0,1)*alpha,
                clip_on=False, transform=t)
            

def get2CTrajectoryDist(track1, track2):
    track1, track2 = track1.copy(), track2.copy()
    head1 = (track1['leftEar'][['x','y']] + track1['rightEar'][['x','y']]) / 2
    head2 = (track2['leftEar'][['x','y']] + track2['rightEar'][['x','y']]) / 2
    track1[('head','x')], track1[('head','y')] = head1.x, head1.y
    track2[('head','x')], track2[('head','y')] = head2.x, head2.y
    track1 = (track1.loc[:,track1.columns.get_level_values(1).isin(['x','y'])]
                    .loc[:,['body','head','tailBase']]
                    .stack(0))
    track2 = (track2.loc[:,track2.columns.get_level_values(1).isin(['x','y'])]
                    .loc[:,['body','head','tailBase']]
                    .stack(0))
    dists = ((track1-track2)**2).sum(axis=1, skipna=False).map(np.sqrt).unstack(-1).mean().mean()
    return dists
    