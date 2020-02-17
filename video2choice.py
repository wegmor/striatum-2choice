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
import moviepy.editor
import moviepy.video.io
import moviepy.video.io.bindings

import analysisTunings
import style
from utils import readSessions, fancyViz

style.set_context()

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
outputFolder = pathlib.Path("video")

if not outputFolder.is_dir():
    outputFolder.mkdir()

class AnimalVideoClip(moviepy.editor.VideoClip): 
    def __init__(self, basePath, session):
        self.videoPath = basePath / sess.meta.video
        self.baseVideo = pims.open(str(self.videoPath)+".avi")
        
        self.fps = self.baseVideo.frame_rate
        self.baseHeight, self.baseWidth, _ = self.baseVideo.frame_shape
        self.sess = session
        self._setupFigure()
        duration = self.baseVideo.get_metadata()["duration"]
        make_frame = lambda t: self._makeFrame(int(np.round(self.fps*t)))
        moviepy.editor.VideoClip.__init__(self, make_frame, False, duration=duration)

    def _setupFigure(self):
        tracking = sess.readTracking()
        self.xCoords = self.baseWidth - tracking[[a for a in tracking.columns if a[1]=='x']].copy()
        self.yCoords = tracking[[a for a in tracking.columns if a[1]=='y']].copy()
        likelihood = tracking[[a for a in tracking.columns if a[1]=='likelihood']].copy()
        self.xCoords[likelihood < 0.9] = np.nan
        self.yCoords[likelihood < 0.9] = np.nan
        self.xCoords[self.xCoords < 10] = np.nan
        self.yCoords[self.yCoords < 10] = np.nan
        lfa = sess.labelFrameActions()
        self.dropCoordinates = pd.DataFrame({
            'alpha': (1-lfa.actionProgress) * lfa.label.str.match("p[LR].+r"),
            'x': 200 + 370*(lfa.label.str[1]=='R'),
            'y': 70 - lfa.actionProgress * 100
        })
        self.mplFig = plt.figure(figsize=(self.baseHeight/100, self.baseWidth/100), dpi=100)
        self.mplAx = self.mplFig.gca()
        self.mplAx.axis("off")
        self.mplFig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.frameIm = self.mplAx.imshow(np.rot90(self.baseVideo.get_frame(0)))
        self.markers, = self.mplAx.plot([], [], '.', ms=15, alpha=.75, color="C1")
        self.bodyLine, = self.mplAx.plot([], [], 'C0', lw=3, alpha=0.5)
        self.neckLine, = self.mplAx.plot([], [], 'C0', lw=3, alpha=0.5)
        self.headLine, = self.mplAx.plot([], [], 'C0', lw=3, alpha=0.5)
        for x, p in [(170, 'pL'), (350, 'pC'), (530, 'pR')]:
            c = np.array(mpl.colors.to_rgb(style.getColor(p)))
            c = 0.25*c + 0.75*np.ones_like(c)
            rect = mpl.patches.Rectangle((x, 10), 60, 40, color=c)
            self.mplAx.add_artist(rect)
        self.drop = None
        
    def _makeFrame(self, i):
        self.frameIm.set_data(np.rot90(self.baseVideo.get_frame(i)))
        self.markers.set_data((self.yCoords.iloc[i], self.xCoords.iloc[i]))
        self.bodyLine.set_data([(self.yCoords.iloc[i, 1], self.yCoords.iloc[i, 4]),
                                (self.xCoords.iloc[i, 1], self.xCoords.iloc[i, 4])])
        self.neckLine.set_data([(self.yCoords.iloc[i, 4], self.yCoords.iloc[i, 2:4].mean()),
                                (self.xCoords.iloc[i, 4], self.xCoords.iloc[i, 2:4].mean())])
        self.headLine.set_data([(self.yCoords.iloc[i, 2], self.yCoords.iloc[i, 3]),
                                (self.xCoords.iloc[i, 2], self.xCoords.iloc[i, 3])])
        if self.drop is not None: self.drop.remove()
        self.drop = fancyViz.drawWaterDrop(self.mplAx, self.dropCoordinates[["x", "y"]].values[i],
                                           -40, facecolor="C0", lw=1, zorder=20,
                                           alpha=self.dropCoordinates.alpha.iloc[i])
        return moviepy.video.io.bindings.mplfig_to_npimage(self.mplFig)

class CalciumVideoClip(moviepy.editor.VideoClip):
    def __init__(self, session, caVideoClip):
        self.baseVideo = pims.open(str(caVideoClip))
        self.fps = self.baseVideo.frame_rate
        self.baseHeight, self.baseWidth, _ = self.baseVideo.frame_shape
        self.sess = session
        self._setupFigure()
        duration = self.baseVideo.get_metadata()["duration"] - 1/self.fps
        make_frame = lambda t: self._makeFrame(int(np.round(self.fps*t)))
        moviepy.editor.VideoClip.__init__(self, make_frame, False, duration=duration)
    
    def markNeurons(self, neuronIds, colors):
        self.selectedNeurons = neuronIds
        self.neuronColors = colors
        roiPeaks = self.sess.readROIs().idxmax(axis=0)
        for n, c in zip(neuronIds, colors):
            x, y = roiPeaks[n]
            x *= 2
            y *= 2
            self.mplAx.fill([y, y+20, y+10, y], [x, x-10, x-20, x], color=c)
        
    def _setupFigure(self):
        self.mplFig = plt.figure(figsize=(self.baseWidth/100, self.baseHeight/100), dpi=100)
        self.mplAx = self.mplFig.gca()
        self.mplAx.axis("off")
        self.mplFig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.frameIm = self.mplAx.imshow(self.baseVideo.get_frame(0))
        
    def _makeFrame(self, i):
        self.frameIm.set_data(self.baseVideo.get_frame(i))
        return moviepy.video.io.bindings.mplfig_to_npimage(self.mplFig)


class TraceVideoClip(moviepy.editor.VideoClip):
    def __init__(self, session, selectedNeurons, colors, width, height, drawStart, drawStop):
        self.sess = session
        self._setupFigure(selectedNeurons, colors, width, height, drawStart, drawStop)
        self.fps = 20
        duration = len(self.deconvTraces) / self.fps
        make_frame = lambda t: self._makeFrame(int(np.round(self.fps*t)))
        moviepy.editor.VideoClip.__init__(self, make_frame, False, duration=duration)
    
    def _setupFigure(self, selectedNeurons, colors, width, height, drawStart, drawStop):
        self.mplFig = plt.figure(figsize=(width/100, height/100), dpi=100)
        self.mplAx = self.mplFig.gca()
        self.mplAx.axis("off")
        self.mplFig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.deconvTraces = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
        lfa = sess.labelFrameActions()
        labels = lfa.loc[drawStart:drawStop, ['label']]
        rewards = labels.loc[labels.label.str.endswith('r').astype('int').diff()==1].index.values
        for n,c in zip(selectedNeurons, colors):
            self.mplAx.vlines(np.arange(drawStart, drawStop),
                              0, self.deconvTraces.loc[drawStart:drawStop, n],
                              lw=2.5, clip_on=False, color=c)
        for l in ['pC','pR','pL']:
            c = np.array(mpl.colors.to_rgb(style.getColor(l)))
            c = 0.25*c + 0.75*np.ones_like(c)
            self.mplAx.fill_between(labels.index.values, 11, -1,              
                                    where=labels['label'].str.slice(0,2) == l,
                                    color=c, lw=0)
        for r in rewards:
            fancyViz.drawWaterDrop(self.mplAx, np.array([r, 9.7]), np.array([4,1]),
                                   facecolor='k')
            self.mplAx.axvline(r, 0, .67, lw=.5, ls='--', color='k')
        transAxes = self.mplAx.transAxes
        self.mplAx.plot([0.5, 0.5], [0, 1], 'k', lw=2, zorder=101, transform=transAxes)
        self.mplAx.plot([0.5], [1], 'k', marker="v", zorder=101, markersize=20, transform=transAxes)
        self.mplAx.plot([0.5], [0], 'k', marker="^", zorder=101, markersize=20, transform=transAxes)
        self.mplAx.fill([0.5, 0.5, 1.0, 1.0], [0, 1, 1, 0], "w", alpha=0.8, zorder=100, transform=transAxes)
        self.mplAx.set_ylim((0,12))
        self.mplAx.set_xlim((0, 20*60))
        self.mplAx.axis('off')
        self.mplAx.set_facecolor('white')
    
    def _makeFrame(self, i):
        self.mplAx.set_xlim(i-300, i+300)
        return moviepy.video.io.bindings.mplfig_to_npimage(self.mplFig)

#%%
ex_session = {'genotype': 'oprm1', 'animal': '5308',
              'date': '190131', 'task': '2choice'}
sess = next(readSessions.findSessions(endoDataPath, **ex_session))
videoFolder = pathlib.Path('/media/emil/mowe-inscopix/videos')
calciumFile = pathlib.Path("/home/emil/2choice/190131_oprm1_5308_example_dff_re.mp4")
selectedNeurons = [106, 112, 0]
colors = [style.getColor(t) for t in ['mC2R', 'pC2L','pL2C']]
startFrame = 12896
stopFrame = startFrame + 2400 - 1

animalClip = AnimalVideoClip(videoFolder, sess)
calciumClip = CalciumVideoClip(sess, calciumFile)
calciumClip.markNeurons(selectedNeurons, colors)
calciumClip = calciumClip.resize(height=animalClip.h)
traceClip = TraceVideoClip(sess, selectedNeurons, colors, animalClip.w + calciumClip.w, 132,
                           startFrame-300, stopFrame+300).subclip(startFrame/20.0, stopFrame/20.0)
animalClip = animalClip.set_pos((0, traceClip.h)).subclip(startFrame/20.0, stopFrame/20.0)
calciumClip = calciumClip.set_pos((animalClip.w, traceClip.h))

compositeClip = moviepy.editor.CompositeVideoClip([traceClip, animalClip, calciumClip],
                                                   size=(traceClip.w, traceClip.h+animalClip.h))
compositeClip.write_videofile(str(outputFolder / "fig3video.mp4"))
