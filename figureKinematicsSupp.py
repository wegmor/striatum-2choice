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
from skimage import exposure
import figurefirst
import style

import analysisKinematicsSupp

style.set_context()
plt.ioff()


#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
    
svgName = "kinematicsSupp.svg"
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}


#%%
endoDataPath = "data/endoData_2019.hdf"
ex_session = {'genotype': 'oprm1', 'animal': '5308', 'date': '190201'}
ex_action = (0,204)
oftSess = next(readSessions.findSessions(endoDataPath, **ex_session, task='openField'))
videoFolder = pathlib.Path('/home/emil/2choice/openFieldVideos')

#%%
open_field_video = pims.open(str(videoFolder / oftSess.meta.video) + ".avi")
tracking = oftSess.readTracking()
segmented = pd.read_pickle("cache/segmentedBehavior.pkl").loc[str(oftSess)]
background = np.median([open_field_video.get_frame(i) for i in tqdm.trange(2000)], axis=0)

chSess = next(readSessions.findSessions(endoDataPath, **ex_session, task='2choice'))
videoFolder = pathlib.Path('/mnt/dmclab/striatum_2choice/')
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
#plt.close()
nNeurons = []
for sess in analysisKinematicsSupp.find2choiceSessionsFollowingOpenField(endoDataPath):
    deconv = sess.readDeconvolvedTraces(zScore=False)
    nNeurons.append((str(sess), deconv.shape[1]))
nNeurons = pd.DataFrame(nNeurons, columns=["sess", "nNeurons"])
nNeurons.set_index("sess", inplace=True)

#fig, axs = plt.subplots(1, 3, sharey=True)
#ax_dict = {'d1': axs[0], 'a2a': axs[1], 'oprm1': axs[2]}
twoChoicePdists = analysisKinematicsSupp.twoChoicePdists(endoDataPath)
openFieldPdists = analysisKinematicsSupp.openFieldPdists(endoDataPath)
cdists = analysisKinematicsSupp.openFieldToTwoChoiceCdists(endoDataPath)
dist_list = [twoChoicePdists, openFieldPdists, cdists]
cols = ["C3", "C4", "C5"]
gts = ["d1", "a2a", "oprm1"]
for i, dist in enumerate(dist_list):
    mean = {gt: np.zeros(20-1) for gt in gts}
    nTot = {gt: 0 for gt in gts}
    for s, g in dist.groupby(level=0):
        bins = pd.cut(g.kinematics_dist, np.linspace(.1, 4, 20))
        binned = g.groupby(bins).mean()
        gt = s.split("_")[0]
        n = nNeurons.loc[s].nNeurons
        ax = layout.axes['kinematicsVsDeconv_'+gt]['axis']
        ax.plot(binned.kinematics_dist, binned.deconv_dist, color=cols[i],
                alpha=np.clip(n/500, 0.1, 1.0), lw=.5)
        mean[gt] += n*binned.deconv_dist
        nTot[gt] += n
    for gt in gts:
        mean[gt] /= nTot[gt]
        ax = layout.axes['kinematicsVsDeconv_'+gt]['axis']
        ax.plot(binned.kinematics_dist, mean[gt], color=cols[i], lw=2)
        
gt_names = {"d1": "D1+", "a2a": "A2A+", "oprm1": "Oprm1+"}
for gt in gts:
    ax = layout.axes['kinematicsVsDeconv_'+gt]['axis']
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.05, 0.12)
    ax.axhline(0, color="k", alpha=0.3, lw=0.5, linestyle="--")
    ax.set_title(gt_names[gt], color=style.getColor(gt))
    ax.set_yticks(np.arange(-0.05, 0.15, 0.05))
    if gt=="d1":
        ax.set_ylabel("ensamble correlation")
    else:
        ax.set_yticklabels([])
    if gt=="a2a":
        ax.set_xlabel("kinematic similarity (Mahalanobis distance)")
    if gt=="oprm1":
        lines = [mpl.lines.Line2D([], [], color=c, label=l) 
                 for c,l in zip(cols, ["open field → open field",
                                       "2-choice → 2-choice",
                                       "open field → 2-choice"])]
        ax.legend(handles=lines, loc=(1.05, 0.4))#'center right')
    sns.despine(ax=ax)


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)