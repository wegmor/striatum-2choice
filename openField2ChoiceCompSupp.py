#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:14:19 2020

@author: mowe
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pims
import tqdm
import cmocean
import seaborn as sns
import scipy.ndimage
from utils import fancyViz, readSessions
import analysisOpenField
from skimage import exposure
import figurefirst
import style

style.set_context()
plt.ioff()


#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")
videoFolder = pathlib.Path('data/')

if not outputFolder.is_dir():
    outputFolder.mkdir()
    
svgName = "oftVsChoiceSupp.svg"
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}


#%% load example session data
ex_session = {'genotype': 'oprm1', 'animal': '5308', 'date': '190201'}
ex_action = (0,204)

oftSess = next(readSessions.findSessions(endoDataPath, **ex_session, task='openField'))
open_field_video = pims.open(str(videoFolder / oftSess.meta.video) + ".avi")
tracking = oftSess.readTracking()
segmented = analysisOpenField.segmentAllOpenField(endoDataPath).loc[str(oftSess)]
background = np.median([open_field_video.get_frame(i) for i in tqdm.trange(2000)], axis=0)

chSess = next(readSessions.findSessions(endoDataPath, **ex_session, task='2choice'))
two_choice_video = pims.open(str(videoFolder / chSess.meta.video) + ".avi")


#%% get example oft turn
start, stop = segmented.loc[ex_action][["startFrame", "stopFrame"]]
frame_ids = list(range(start, stop+1, 5))
frames = np.array([open_field_video.get_frame(i) for i in frame_ids])
coords = tracking.loc[start:stop]


#%% plot first oft frame
ax = layout.axes['trajectoryIllustration','openField']['axis']

ax.imshow(exposure.adjust_log(frames[0],1.3))
xy = coords.iloc[0]
ax.plot([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
        [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
        color='yellow', lw=mpl.rcParams['axes.linewidth'], zorder=1)
ax.scatter([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
           [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
           color='yellow', zorder=1, marker='.')
ax.set_xlim((120,709))
ax.set_ylim((590,0))
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

cmap = sns.cubehelix_palette(start=1.4, rot=.8*np.pi, light=.75, as_cmap=True)
for i in range(len(frames)):
    frame = frames[i].mean(axis=-1) + i*255 - 255*len(frames)/2 + 256
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


#%% plot 2 choice frame
chTracking = chSess.readTracking()
n = 3933
frame = two_choice_video.get_frame(n)

ax = layout.axes['trajectoryIllustration','choice']['axis']
ax.set_ylim((750,0))
ax.set_xlim((55,800))
t = ax.transData
tr = mpl.transforms.Affine2D().rotate_deg_around(frame.shape[1]/2, frame.shape[0]/2, -90) + t

ax.imshow(exposure.adjust_gamma(frame, .7), transform=tr)
xy = chTracking.loc[n]
ax.plot([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
        [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
        color='yellow', lw=mpl.rcParams['axes.linewidth'], zorder=1,
        transform=tr)
ax.scatter([xy.tailBase.x, xy.body.x, 0.5*(xy.leftEar.x + xy.rightEar.x)],
           [xy.tailBase.y, xy.body.y, 0.5*(xy.leftEar.y + xy.rightEar.y)],
           color='yellow', zorder=1, marker='.', transform=tr)
ax.axis('off')


#%%



#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)