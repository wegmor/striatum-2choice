import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import h5py
import pathlib
import cmocean
import tqdm
import pims

import analysisTunings
import style
from utils import readSessions, fancyViz

style.set_context()

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path("video")

if not outputFolder.is_dir():
    outputFolder.mkdir()
    
ex_session = {'genotype': 'oprm1', 'animal': '5308',
              'date': '190131', 'task': '2choice'}
sess = next(readSessions.findSessions(endoDataPath, **ex_session))

animalVideoPath = pathlib.Path('/media/emil/mowe-inscopix/videos') / sess.meta.video
animalVideo = pims.open(str(animalVideoPath)+".avi")


tracking = sess.readTracking()
xCoords = animalVideo.frame_shape[1] - tracking[[a for a in tracking.columns if a[1]=='x']].copy()
yCoords = tracking[[a for a in tracking.columns if a[1]=='y']].copy()
likelihood = tracking[[a for a in tracking.columns if a[1]=='likelihood']].copy()
xCoords[likelihood < 0.9] = np.nan
yCoords[likelihood < 0.9] = np.nan
xCoords[xCoords < 10] = np.nan
yCoords[yCoords < 10] = np.nan

lfa = sess.labelFrameActions()

fig = plt.figure(figsize=(7.50, 8.72), dpi=100)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
animalIm = plt.imshow(np.rot90(animalVideo.get_frame(0)))
plt.axis("off")
markers, = plt.plot(yCoords.iloc[0], xCoords.iloc[0], '.', ms=15, alpha=.75, color="C1")
bodyLine, = plt.plot((yCoords.iloc[0, 1], yCoords.iloc[0, 4]),
                     (xCoords.iloc[0, 1], xCoords.iloc[0, 4]), lw=3, alpha=0.5)
neckLine, = plt.plot((yCoords.iloc[0, 4], yCoords.iloc[0, 2:4].mean()),
                     (xCoords.iloc[0, 4], xCoords.iloc[0, 2:4].mean()), "C0", lw=3, alpha=0.5)
headLine, = plt.plot((yCoords.iloc[0, 2], yCoords.iloc[0, 3]),
                     (xCoords.iloc[0, 2], xCoords.iloc[0, 3]), "C0", lw=3, alpha=0.5)
ax = plt.gca()
for x, p in [(170, 'pL'), (350, 'pC'), (530, 'pR')]:
    ax.add_artist(mpl.patches.Rectangle((x, 10), 60, 40, color=style.getColor(p)))
    
lfa["alpha"] = (1-lfa.actionProgress) * lfa.label.str.match("p[LR].+r")
lfa["dropX"] = 200 + 370*(lfa.label.str[1]=='R')
lfa["dropY"] = 70 - lfa.actionProgress * 100
drop = None

def animate(i):
    global drop
    animalIm.set_data(np.rot90(animalVideo.get_frame(i)))
    markers.set_data((yCoords.iloc[i], xCoords.iloc[i]))
    bodyLine.set_data([(yCoords.iloc[i, 1], yCoords.iloc[i, 4]),
                       (xCoords.iloc[i, 1], xCoords.iloc[i, 4])])
    neckLine.set_data([( yCoords.iloc[i, 4], yCoords.iloc[i, 2:4].mean()),
                       (xCoords.iloc[i, 4], xCoords.iloc[i, 2:4].mean())])
    headLine.set_data([(yCoords.iloc[i, 2], yCoords.iloc[i, 3]),
                       (xCoords.iloc[i, 2], xCoords.iloc[i, 3])])
    if drop is not None: drop.remove()
    drop = fancyViz.drawWaterDrop(ax, lfa[["dropX", "dropY"]].values[i], -40,
                                  facecolor="C0", lw=1, alpha=lfa.alpha.iloc[i], 
                                  zorder=20)
anim = mpl.animation.FuncAnimation(fig, animate, frames=tqdm.trange(12896, 12896+2400))
anim.save(outputFolder / "behaviorFromFig3.mp4", fps=20, dpi=100)


#%%
roiPeaks = sess.readROIs().idxmax(axis=0)
caHeight = 872
caWidth = 600*(872/504)
fig = plt.figure(figsize=(caWidth/100, caHeight/100), dpi=100)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
caVideo = pims.open("/home/emil/2choice/190131_oprm1_5308_example_dff_re.mp4")
#F = np.array([caVideo.get_frame(i) for i in range(2400)]).mean(axis=0)
caIm = plt.imshow(caVideo.get_frame(0))
plt.axis("off")
selectedNeurons = [106, 112, 0]
colors = [style.getColor(t) for t in ['mC2R', 'pC2L','pL2C']]
for n,c in zip(selectedNeurons, colors):
    x, y = roiPeaks[n]
    x *= 2
    y *= 2
    plt.fill([y, y+20, y+10, y], [x, x-10, x-20, x], color=c)

def animateCa(i):
    caIm.set_data((caVideo.get_frame(i)))

anim = mpl.animation.FuncAnimation(fig, animateCa, frames=tqdm.trange(2400))
anim.save(outputFolder / "caBetterFromFig3.mp4", fps=20, dpi=100)

#%%
#caVideoPath = pathlib.Path('/media/emil/mowe-inscopix/calcium_videos/') / (str(sess) + ".nwb")
#caVideoFile = h5py.File(caVideoPath, 'r')

#def getCaFrame(caVideoFile, i):
#    recordings = caVideoFile["analysis"]
#    j = 0
#    for recording in recordings:
#        data = caVideoFile["analysis/{}/data".format(recording)]
#        if i < j+len(data):
#            return data[i-j]
#        j += len(data)
#    raise IndexError("Frame {} is larger than video length ({})".format(i, j))

#%%
fig = plt.figure(figsize=(17.88, 1.338), dpi=100)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
start = 12896
stop = start + 2400
#Extend trace 15s in each direction
start -= 300
stop += 300
labels = lfa.loc[start:stop, ['label']]
rewards = labels.loc[labels.label.str.endswith('r').astype('int').diff()==1].index.values
for n,c in zip(selectedNeurons, colors):
    plt.vlines(np.arange(start, stop),
               0, deconv.loc[start:stop, n],
               lw=2.5, clip_on=False, color=c)
for l in ['pC','pR','pL']:
    plt.fill_between(labels.index.values, 11, -1,              
                     where=labels['label'].str.slice(0,2) == l,
                     color=style.getColor(l), lw=0, alpha=.25)
for r in rewards:
    fancyViz.drawWaterDrop(plt.gca(), np.array([r, 9.7]), np.array([4,1]),
                           facecolor='k')
    plt.axvline(r, 0, .67, lw=.5, ls='--', color='k')
transAxes = plt.gca().transAxes
plt.plot([0.5, 0.5], [0, 1], 'k', lw=2, zorder=101, transform=transAxes)
plt.plot([0.5], [1], 'k', marker="v", zorder=101, markersize=20, transform=transAxes)
plt.plot([0.5], [0], 'k', marker="^", zorder=101, markersize=20, transform=transAxes)
plt.fill([0.5, 0.5, 1.0, 1.0], [0, 1, 1, 0], "w", alpha=0.9, zorder=100, transform=transAxes)
plt.gca().set_ylim((0,12))
plt.gca().set_xlim((start, start+20*60))
plt.axis('off')
def animateTrace(i):
    plt.xlim(start+i, start+i+600)
anim = mpl.animation.FuncAnimation(fig, animateTrace, frames=tqdm.trange(2400))
anim.save(outputFolder / "traceFromFig3_v2.mp4", fps=20, dpi=100)


#%%
import moviepy.editor

caClip = moviepy.editor.VideoFileClip(str(outputFolder / "caBetterFromFig3.mp4"))
behaviorClip = moviepy.editor.VideoFileClip(str(outputFolder / "behaviorFromFig3.mp4"))
traceClip = moviepy.editor.VideoFileClip(str(outputFolder / "traceFromFig3_v2.mp4"))

compositeClip = moviepy.editor.CompositeVideoClip([traceClip,
                                                   behaviorClip.set_pos((0,132)),
                                                   caClip.set_pos((750,132))],
                                                   size=(1788, 132+872))import moviepy.editor

caClip = moviepy.editor.VideoFileClip(str(outputFolder / "caBetterFromFig3.mp4"))
behaviorClip = moviepy.editor.VideoFileClip(str(outputFolder / "behaviorFromFig3.mp4"))
traceClip = moviepy.editor.VideoFileClip(str(outputFolder / "traceFromFig3_v2.mp4"))

compositeClip = moviepy.editor.CompositeVideoClip([traceClip,
                                                   behaviorClip.set_pos((0,132)),
                                                   caClip.set_pos((750,132))],
                                                   size=(1788, 132+872))
compositeClip.write_videofile(str(outputFolder / "fig3Combined.mp4"), fps=20, audio=False)
compositeClip.write_videofile(str(outputFolder / "fig3Combined.mp4"), fps=20, audio=False)