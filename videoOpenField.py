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

import analysisOpenField
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
        tracking = self.sess.readTracking()
        self.xCoords = tracking[[a for a in tracking.columns if a[1]=='x']].copy()
        self.yCoords = tracking[[a for a in tracking.columns if a[1]=='y']].copy()
        likelihood = tracking[[a for a in tracking.columns if a[1]=='likelihood']].copy()
        self.xCoords[likelihood < 0.9] = np.nan
        self.yCoords[likelihood < 0.9] = np.nan
        self.xCoords[self.xCoords < 10] = np.nan
        self.yCoords[self.yCoords < 10] = np.nan
        
        self.mplFig = plt.figure(figsize=(self.baseWidth/100, self.baseHeight/100), dpi=100)
        self.mplAx = self.mplFig.gca()
        self.mplAx.axis("off")
        self.mplFig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.frameIm = self.mplAx.imshow(self.baseVideo.get_frame(0))
        self.markers, = self.mplAx.plot([], [], '.', ms=5, alpha=.75, color="C1")
        #self.bodyLine, = self.mplAx.plot([], [], 'C0', lw=3, alpha=0.5)
        
        segs = analysisOpenField.segmentAllOpenField(endoDataPath)
        segs = segs.loc[str(self.sess)]
        stopFrame = segs.stopFrame[-1]
        self.behaviors = segs.set_index("startFrame").behavior
        self.behaviors = self.behaviors.reindex(pd.RangeIndex(0, stopFrame),
                                                method="ffill")
        behavior = self.behaviors.loc[0]
        self.behaviorLabel = self.mplAx.text(self.xCoords.iloc[0, 0],
                                             self.yCoords.iloc[0, 0]-20,
                                             behavior,
                                             color=style.getColor(behavior),
                                             ha="center",
                                             va="bottom",
                                             fontsize=14,
                                             fontweight="bold")
        
        #self.neckLine, = self.mplAx.plot([], [], 'C0', lw=3, alpha=0.5)
        #self.headLine, = self.mplAx.plot([], [], 'C0', lw=3, alpha=0.5)
        
    def _makeFrame(self, i):
        self.frameIm.set_data(self.baseVideo.get_frame(i))
        self.markers.set_data((self.xCoords.iloc[i], self.yCoords.iloc[i]))
        #self.bodyLine.set_data([(self.xCoords.iloc[i, 0], self.xCoords.iloc[i, 1]),
        #                        (self.yCoords.iloc[i, 0], self.yCoords.iloc[i, 1])])
        self.behaviorLabel.set_x(self.xCoords.iloc[i, 0])
        self.behaviorLabel.set_y(self.yCoords.iloc[i, 0]-20)
        self.behaviorLabel.set_text(self.behaviors.loc[i])
        self.behaviorLabel.set_color(style.getColor(self.behaviors.loc[i]))
        return moviepy.video.io.bindings.mplfig_to_npimage(self.mplFig)

    
ex_session = {'genotype': 'oprm1', 'animal': '5308',
              'date': '190224', 'task': 'openField'}
sess = next(readSessions.findSessions(endoDataPath, **ex_session))
videoFolder = pathlib.Path('/home/emil/2choice/openFieldVideos')
#calciumFile = pathlib.Path("/home/emil/2choice/190131_oprm1_5308_example_dff_re.mp4")
tuningData = analysisOpenField.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005
#queryStr = " & ".join(["{}=='{}'".format(*v) for v in ex_session.items()])
#ex_tunings = tuningData.query(queryStr)


#selectedNeurons = ex_tunings.set_index("neuron").groupby("action").tuning.idxmax()
order = ["leftTurn", "rightTurn", "running"]

colors = [style.getColor(t) for t in order]
startFrame = 3200
stopFrame = startFrame + 2400 - 1

animalClip = AnimalVideoClip(videoFolder, sess)
#calciumClip = CalciumVideoClip(sess, calciumFile)
#calciumClip.markNeurons(selectedNeurons, colors)
#calciumClip = calciumClip.resize(height=animalClip.h)
#traceClip = TraceVideoClip(sess, selectedNeurons, colors, animalClip.w + calciumClip.w, 132,
#                           startFrame-300, stopFrame+300).subclip(startFrame/20.0, stopFrame/20.0)
#animalClip = animalClip.set_pos((0, traceClip.h)).subclip(startFrame/20.0, stopFrame/20.0)
#calciumClip = calciumClip.set_pos((animalClip.w, traceClip.h))

#compositeClip = moviepy.editor.CompositeVideoClip([traceClip, animalClip, calciumClip],
#                                                   size=(traceClip.w, traceClip.h+animalClip.h))
#compositeClip.write_videofile(str(outputFolder / "fig3video.mp4"))
animalClip.subclip(startFrame/20.0, stopFrame/20.0).write_videofile(str(outputFolder / "fig1video.mp4"))
