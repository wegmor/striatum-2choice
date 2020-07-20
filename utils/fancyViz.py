import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
import scipy.ndimage
import skimage.measure
import matplotlib.backends.backend_pdf
import os
import PIL
import numbers
from deprecated import deprecated

from . import fancyVizUtils, trackingGeometryUtils

class IntensityPlot:

    def setMask(self, mask):
        self.mask = mask
    
    def draw(self, trace, ax=None, **kwargs):
        self.clearBuffer()
        self.addTraceToBuffer(trace)
        self.drawBuffer(ax, **kwargs)
    
    def addTraceToBuffer(self, trace):
        trace = np.array(trace).astype(np.float)
        if len(trace) != len(self.coordinates):
            raise ValueError("The trace contains {} frames but the coordinates {}".format(len(trace), len(self.coordinates)))
        fancyVizUtils.integerHistogram(self.coordinates[self.mask], trace[self.mask],
                                       self.valueCanvas, self.normCanvas)
    
    def drawBuffer(self, ax=None, **kwargs):
        if ax is not None: plt.sca(ax)
        values = scipy.ndimage.gaussian_filter(self.valueCanvas, self.smoothing)
        norm   = scipy.ndimage.gaussian_filter(self.normCanvas, self.smoothing)
        normalized = np.divide(values, norm, out=np.zeros_like(values), where=norm!=0)
        self._drawSchema(normalized, norm*10, **kwargs)
        self.clearBuffer()
        
class SchematicIntensityPlot(IntensityPlot):
    
    def __init__(self, session=None, saturation=1.0, smoothing=4, linewidth=2, waterDrop=True,
                 portRadius=0.4, splitCenter=True, splitReturns=True):
        self.saturation = saturation
        self.smoothing = smoothing
        self.linewidth = linewidth
        self.waterDrop = waterDrop
        self.portRadius = portRadius
        self.splitCenter = splitCenter
        self.splitReturns = splitReturns
        self.mask = slice(None, None) #No mask, use all values
        self.clearBuffer()
        if session is not None:
            self.setSession(session)
    
    def clearBuffer(self):
        self.valueCanvas = np.zeros((251,501), np.float64)
        self.normCanvas = np.zeros((251,501), np.float64)
    
    def _drawSchema(self, im, alpha, ax=None, **kwargs):
        '''Internal function, do not call directly.'''
        imshowWithAlpha(im, alpha, self.saturation, origin="lower",
                        extent=(-5,5,-2.5,2.5), zorder=-100000, **kwargs)
        plt.xlim(-5.5, 5.5)
        plt.ylim(-2.75, 2.5)
        r = self.portRadius
        lw = self.linewidth
        
        if self.splitCenter:
            drawRoundedRect(plt.gca(), ( 0.05, -1), 0.7, 2, [0, 0, r, r], fill=False, lw=lw, zorder=-10000, edgecolor="k")
            drawRoundedRect(plt.gca(), (-0.75, -1), 0.7, 2, [r, r, 0, 0], fill=False, lw=lw, zorder=-10000, edgecolor="k")
        else:
            drawRoundedRect(plt.gca(), (-0.75, -1), 1.5, 2, [r, r, r, r], fill=False, lw=lw, zorder=-10000, edgecolor="k")

        drawRoundedRect(plt.gca(), (-4.75, -1),  0.7, 1.2, [r, 0, 0, 0], fill=False, lw=lw, zorder=-10000, edgecolor="k")
        drawRoundedRect(plt.gca(), (-3.95, -1),  0.7, 1.2, [0, 0, 0, r], fill=False, lw=lw, zorder=-10000, edgecolor="k")
        drawRoundedRect(plt.gca(), (-4.75, 0.3), 1.5, 0.7, [0, r, r, 0], fill=False, lw=lw, zorder=-10000, edgecolor="k")

        drawRoundedRect(plt.gca(), (3.25, -1), 0.7, 1.2,  [r, 0, 0, 0], fill=False, lw=lw, zorder=-10000, edgecolor="k")
        drawRoundedRect(plt.gca(), (4.05, -1), 0.7, 1.2,  [0, 0, 0, r], fill=False, lw=lw, zorder=-10000, edgecolor="k")
        drawRoundedRect(plt.gca(), (3.25, 0.3), 1.5, 0.7, [0, r, r, 0], fill=False, lw=lw, zorder=-10000, edgecolor="k")
        
        xx = np.linspace(-1,1)
        yy = 1-xx*xx
        plt.plot(xx*1.8+2.2, yy+1,'k', lw=lw, zorder=-10000)
        plt.plot(xx*-1.8-2.2,yy+1,'k', lw=lw, zorder=-10000)
        plt.plot(xx*2+2,-yy-1,'k', lw=lw, zorder=-10000)
        plt.plot(xx*-2-2,-yy-1,'k', lw=lw, zorder=-10000)
        drawArrowHead(plt.gca(), (xx[-5]*1.8+2.2, yy[-5]+1), (xx[-1]*1.8+2.2, yy[-1]+1), facecolor="k", edgecolor="k")
        drawArrowHead(plt.gca(), (xx[-5]*-1.8-2.2, yy[-5]+1), (xx[-1]*-1.8-2.2, yy[-1]+1), facecolor="k", edgecolor="k")
        drawArrowHead(plt.gca(), (xx[4]*2+2, -yy[4]-1), (xx[0]*2+2, -yy[0]-1), facecolor="k", edgecolor="k")
        drawArrowHead(plt.gca(), (xx[4]*-2-2, -yy[4]-1), (xx[0]*-2-2, -yy[0]-1), facecolor="k", edgecolor="k")
        if self.waterDrop:
            drawWaterDrop(plt.gca(), np.array([-2.75, -0.5]), 0.3, lw=lw)
            drawWaterDrop(plt.gca(), np.array([ 2.75, -0.5]), 0.3, lw=lw)
            if self.splitReturns:
                drawWaterDrop(plt.gca(), np.array([-4.6, -1.5]), 0.3, True, lw=lw)
                drawWaterDrop(plt.gca(), np.array([ 4.6, -1.5]), 0.3, True, lw=lw)
            else:
                drawWaterDrop(plt.gca(), np.array([-5.25, -0.5]), 0.3, True, lw=lw)
                drawWaterDrop(plt.gca(), np.array([ 5.25, -0.5]), 0.3, True, lw=lw)
        plt.axis("off")
        
    def setSession(self, session):
        inclRew = 'returns' if self.splitReturns else "sidePorts"
        lfa = session.labelFrameActions(reward=inclRew, switch=False,
                                        splitCenter=self.splitCenter)
        schematicCoord = fancyVizUtils.taskSchematicCoordinates(lfa)*50
        schematicCoord.x += 250
        schematicCoord.y += 125
        self.coordinates = schematicCoord.values
        
class TrackingIntensityPlot(IntensityPlot):
    def __init__(self, session=None, smoothing=10, saturation=1.0, portsUp=False, drawBg=True):
        self.smoothing = smoothing
        self.saturation = saturation
        self.portsUp = portsUp
        self.drawBg = drawBg
        self.canvasSize = None
        self.mask = slice(None, None) #No mask, use all values
        self.setSession(session)
        self.clearBuffer()
    
    def setSession(self, session):
        backgroundFolder = os.path.dirname(__file__) + "/video_backgrounds/"
        if session.meta.task == "openField":
            if session.meta.date == "190224":
                self.background = backgroundFolder + "background_open_field_2019_second_camera.png"
            else:
                self.background = backgroundFolder + "background_open_field_2019_first_camera.png"
        else:
            if session.meta.cohort == "2018":
                self.background = backgroundFolder + "background_2choice_2018.png"
            else:
                self.background = backgroundFolder + "background_2choice_2019.png"
        self.backgroundIm = PIL.Image.open(self.background)
        if self.canvasSize is None:
            self.canvasSize = self.backgroundIm.size[::-1]
        elif self.canvasSize != self.backgroundIm.size[::-1]:
            raise ValueError("Cannot change to session {}, the video dimensions do not match.".format(session))
        tracking = session.readTracking()
        headCoordinates = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
        likelihood = tracking[[("leftEar", "likelihood"),
                               ("rightEar", "likelihood"),
                               ("tailBase", "likelihood")]].min(axis=1)
        self.coordinates = headCoordinates.values
        self.coordinates[likelihood.values < 0.9, :] = np.nan
        self.wallCorners = session.getWallCorners()
        self.task = session.meta.task
        
    def clearBuffer(self):
        self.valueCanvas = np.zeros(self.canvasSize, np.float64)
        self.normCanvas = np.zeros(self.canvasSize, np.float64)
            
    def _drawSchema(self, im, alpha):
        if self.drawBg:
            bgIm = self.backgroundIm
            if self.portsUp: bgIm = np.rot90(bgIm)
            plt.imshow(bgIm, alpha=0.5, cmap="gray")
        if self.portsUp:
            im = np.rot90(im)
            alpha = np.rot90(alpha)
        imshowWithAlpha(im, alpha, self.saturation)
        if not self.drawBg: # TODO: 2018 cohort returns a list, not a pandas Series 
            corners = self.wallCorners
            corners_x = corners[corners.index.get_level_values(1) == 'x'].values
            corners_y = corners[corners.index.get_level_values(1) == 'y'].values
            if self.portsUp:
                corners_x, corners_y = corners_y, im.shape[0] - corners_x
            corners_x = np.append(corners_x, corners_x[0])
            corners_y = np.append(corners_y, corners_y[0])
            plt.plot(corners_x, corners_y, 'k', lw=0.5)
            #drawRoundedRect(ax, position, width, height, radius, **kwargs):
            if self.task in ("2choice", "forcedAlternation", "2choiceAgain"):
                if self.portsUp:
                    left = np.min(corners_x)
                    right = np.max(corners_x)
                    top = np.min(corners_y)
                    width = right - left
                    s = width/7
                    for x in left+s*np.array([1, 3, 5]):
                        drawRoundedRect(plt.gca(), (x, top-s), s, s, [s/4, 0, 0, s/4],
                                        fill=False, edgecolor="k")
                else:
                    left = np.min(corners_y)
                    right = np.max(corners_y)
                    top = np.max(corners_x)
                    width = right - left
                    s = width/7
                    for y in left+s*np.array([1, 3, 5]):
                        drawRoundedRect(plt.gca(), (top, y), s, s, [0, 0, s/4, s/4],
                                        fill=False, edgecolor="k")
        plt.axis("off")
        
        
class BodyDirectionPlot(IntensityPlot):
    def __init__(self, block=None, smoothing=6, positionFilter=None):
        self.smoothing = smoothing
        self.canvasSize = (301, 301)
        if block is not None:
            self.setDefaultCoordinates(block, positionFilter)
            
    def draw(self, trace, saturation=0.5, ax=None):
        IntensityPlot.draw(self, trace, saturation, ax, extent=(-1.5,1.5,-1.5,1.5))
        phi = np.linspace(-np.pi, np.pi, 100)
        plt.plot(np.cos(phi), np.sin(phi), 'k')
        plt.axis("off")
        plt.axis("equal")
        
    def setDefaultCoordinates(self, block, positionFilter=None):
        tracking = block.readTracking()
        headCoordinates = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
        likelihood = tracking[[("leftEar", "likelihood"),
                           ("rightEar", "likelihood"),
                           ("tailBase", "likelihood")]].min(axis=1)
        mask = likelihood.values>0.9
        if positionFilter is not None:
            mask = np.logical_and(mask, headCoordinates.eval(positionFilter).values)
        bodyDir = trackingGeometryUtils.calcBodyDirection(tracking)
        projSpeed  = trackingGeometryUtils.calcProjectedSpeed(tracking)
        bodyDirRadius = np.clip(1 + projSpeed / 50.0, 0.5, 1.5)
        bodyDirCoord = np.vstack([np.cos(bodyDir)*100*bodyDirRadius+150,
                                  np.sin(bodyDir)*100*bodyDirRadius+150]).T
        self.setCoordinates(bodyDirCoord, mask)
        
class BodyTurnPlot(IntensityPlot):
    
    def __init__(self, session=None, smoothing=6, saturation=1.0, lw=0.5):
        self.smoothing = smoothing
        self.saturation = saturation
        self.lw = lw
        self.canvasSize = None
        self.mask = slice(None, None) #No mask, use all values
        self.setSession(session)
        self.clearBuffer()
    
    def clearBuffer(self):
        self.valueCanvas = np.zeros((301, 301), np.float64)
        self.normCanvas = np.zeros((301, 301), np.float64)
        
    def _drawSchema(self, im, alpha):
        imshowWithAlpha(im, alpha, self.saturation, extent=(-1.5, 1.5, -1.5, 1.5))
        drawArcArrow(1, -0.1, -2, lw=self.lw)
        drawArcArrow(1 , 0.1, 2, lw=self.lw)
        plt.axis("off")
        plt.axis("equal")
        
    def setSession(self, session):
        tracking = session.readTracking(inCm=True)
        bodyVec = tracking.body - tracking.tailBase
        bodyDir = -np.arctan2(bodyVec.y, bodyVec.x).rename("bodyDirection")
        turningSpeed = trackingGeometryUtils.angleDiff(bodyDir.shift(1), bodyDir)
        likelihood = tracking[[("body", "likelihood"),
                               ("tailBase", "likelihood")]].min(axis=1)
        turningSpeed[turningSpeed>np.pi/6] = np.nan
        self.coordinates = np.vstack([np.cos(turningSpeed*5)*100+150,
                                      np.sin(turningSpeed*5)*100+150]).T
        self.coordinates[likelihood.values < 0.9, :] = np.nan
        self.coordinates[likelihood.shift(1).values < 0.9, :] = np.nan
        
class HeadTurnPlot(IntensityPlot):
    def __init__(self, block=None, smoothing=6, positionFilter=None):
        self.smoothing = smoothing
        self.canvasSize = (301, 151)
        if block is not None:
            self.setDefaultCoordinates(block, positionFilter)
            
    def draw(self, trace, saturation=0.5, ax=None):
        IntensityPlot.draw(self, trace, saturation, ax, extent=(0,1.5,-1.5,1.5))
        phi = np.linspace(-0.5*np.pi,0.5*np.pi,100)
        plt.plot(np.cos(phi), np.sin(phi), 'k')
        offs = np.pi*2 / 20 * 0.5
        drawArcArrow(1 + offs, -0.1, -1.4)
        drawArcArrow(1 + offs, 0.1, 1.4)
        drawArcArrow(1 - offs, -1.4, -0.25)
        drawArcArrow(1 - offs, 1.4,  0.25)
        plt.plot([0,1],[0,0], 'k--')
        plt.axis("equal")
        plt.axis("off")
        
    def setDefaultCoordinates(self, block, positionFilter=None):
        tracking = block.readTracking()
        headCoordinates = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
        likelihood = tracking[[("leftEar", "likelihood"),
                           ("rightEar", "likelihood"),
                           ("tailBase", "likelihood")]].min(axis=1)
        mask = likelihood.values>0.9
        if positionFilter is not None:
            mask = np.logical_and(mask, headCoordinates.eval(positionFilter).values)
        headDir    = trackingGeometryUtils.calcHeadDirections(tracking)
        headDirRadius = np.clip(1 + np.sign(headDir)*headDir.diff()*0.5, 0.5, 1.5)
        headDirCoord = np.vstack([np.cos(headDir)*100*headDirRadius,
                                  np.sin(headDir)*100*headDirRadius+150]).T
        self.setCoordinates(headDirCoord, mask)
        
class GazePointPlot(IntensityPlot):
    def __init__(self, block=None, smoothing=6, positionFilter=None):
        self.smoothing = smoothing
        self.canvasSize = (304, 400)
        if block is not None:
            self.setDefaultCoordinates(block, positionFilter)
            
    def draw(self, trace, saturation=0.5, ax=None):
        IntensityPlot.draw(self, trace, saturation, ax)
        plt.axis("off")
        
    def setDefaultCoordinates(self, block, positionFilter=None):
        tracking = block.readTracking()
        headCoordinates = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
        likelihood = tracking[[("leftEar", "likelihood"),
                           ("rightEar", "likelihood"),
                           ("tailBase", "likelihood")]].min(axis=1)
        mask = likelihood.values>0.9
        if positionFilter is not None:
            mask = np.logical_and(mask, headCoordinates.eval(positionFilter).values)
        gazePoint  = trackingGeometryUtils.calcGazePoint(tracking, block.getWallCorners())
        self.setCoordinates(gazePoint.values, mask)
        
class TimePlot(IntensityPlot):
    def __init__(self, block=None, smoothing=1.5, positionFilter=None):
        self.smoothing = smoothing
        self.canvasSize = (21, 400)
        if block is not None:
            if positionFilter is None:
                positionFilter = "x<{}".format(block.getWallCorners()[2]-70)
            self.setDefaultCoordinates(block, positionFilter)
            
    def draw(self, trace, saturation=0.5, ax=None, xlabel="Time [minutes]"):
        IntensityPlot.draw(self, trace, saturation, ax, extent=(0,len(trace),0,20))
        sns.despine(ax=ax, left=True, bottom=True)
        plt.xticks(60*20*np.arange(0,61,15), ["%dm"%t for t in np.arange(0,61,15)])
        plt.yticks((4,16), ("Other area", "Task area"))
        plt.xlabel(xlabel)
        plt.axis("auto")
        #plt.axis("off")
        
    def setDefaultCoordinates(self, block, positionFilter):
        tracking = block.readTracking()
        headCoordinates = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
        likelihood = tracking[[("leftEar", "likelihood"),
                               ("rightEar", "likelihood"),
                               ("tailBase", "likelihood")]].min(axis=1)
        mask = likelihood.values>0.9
        xx = np.linspace(0,400,len(mask),endpoint=False)
        yy = 4.0 + 12.0*headCoordinates.eval(positionFilter).values
        self.setCoordinates(np.vstack((xx,yy)).T, mask)
        
class ReturnBoutsPlot(IntensityPlot):
    def __init__(self, block=None, smoothing=2):
        self.smoothing = smoothing
        self.canvasSize = (100, 50)
        if block is not None:
            self.setDefaultCoordinates(block)
            
    def draw(self, trace, saturation=0.5, ax=None):
        IntensityPlot.draw(self, trace, saturation, ax)
        sns.despine(ax=ax, left=True, bottom=True)
        plt.xticks([])
        plt.yticks((20,40,60), ("Left (%d)" % self.boutCount.loc["L"],
                                "Center (%d)" % self.boutCount.loc["C"],
                                "Right (%d)" % self.boutCount.loc["R"]))
        plt.gca().yaxis.set_ticks_position("right")
        plt.axis("auto")
        
    def setDefaultCoordinates(self, block):
        tracking = block.readTracking()
        wallCorners = block.getWallCorners()
        returnBouts = trackingGeometryUtils.findReturnBouts(tracking, wallCorners)
        coords = fancyVizUtils.returnBoutsCoordinates(returnBouts, len(tracking))
        self.boutCount = returnBouts.port.value_counts().reindex(index=["L", "C", "R"], fill_value=0)
        self.setCoordinates(coords.values, np.ones(len(tracking), np.bool_))
        
        
class BlockActionsIntensityPlot(IntensityPlot):
    def __init__(self, block=None, smoothing=1.5):
        self.smoothing = smoothing
        self.canvasSize = (251, 601)
        if block is not None:
            self.setDefaultCoordinates(block)
            
    def draw(self, trace, saturation=0.5 ,ax=None):
        IntensityPlot.draw(self, trace, saturation, ax, origin="lower",
                           extent=(-6,6,-2.5,2.5), zorder=-100000)
        leftRect = matplotlib.patches.Rectangle((-5.5, 0.5), 5, 1, fill=False)#, alpha=.5)
        rightRect = matplotlib.patches.Rectangle((0.5, -1.5), 5, 1, fill=False)#, alpha=.5)
        plt.gca().add_artist(leftRect)
        plt.gca().add_artist(rightRect)
        plt.vlines(0.5+5*scipy.stats.geom.cdf(np.arange(1,100,1), 0.05),-1.5,-0.5, lw=0.5, alpha=0.25)
        plt.vlines(-0.5-5*scipy.stats.geom.cdf(np.arange(1,100,1), 0.05),0.5, 1.5, lw=0.5, alpha=0.25)
        plt.axis("off")
        phi = np.linspace(0,np.pi/2,100)
        for i in (0,5):
            plt.plot((i+1)*np.cos(phi)-0.5, (i*0.2+1)*np.sin(phi)-0.5, 'k', lw=0.5)
            plt.plot((i+1)*-np.sin(phi)+0.5, (i*0.2+1)*-np.cos(phi)+0.5, 'k', lw=0.5)
        plt.axis("equal")
        
    def setDefaultCoordinates(self, block):
        sensorValues = block.readSensorValues()
        blockActions = fancyVizUtils.findBlockActions(sensorValues, timeout=40)
        blockCoord = 50*fancyVizUtils.blockActionCoordinates(blockActions,
                                                                 len(sensorValues))
        blockCoord.x += 300
        blockCoord.y += 125
        self.setCoordinates(blockCoord.values, np.ones(len(blockCoord), np.bool_))

class WallAnglePlot(IntensityPlot):
    
    def __init__(self, session=None, smoothing=12, saturation=1.0, lw=0.5):
        self.smoothing = smoothing
        self.saturation = saturation
        self.lw = lw
        self.mask = slice(None, None) #No mask, use all values
        self.setSession(session)
        self.clearBuffer()
    
    def clearBuffer(self):
        self.valueCanvas = np.zeros((301, 301), np.float64)
        self.normCanvas = np.zeros((301, 301), np.float64)
        
    def _drawSchema(self, im, alpha):
        
        imshowWithAlpha(im, alpha, self.saturation, extent=(-5, 5, -5, 5), origin="lower")
        mouseIcon = PIL.Image.open(os.path.dirname(__file__) + "/mouseIcon.png")
        plt.imshow(mouseIcon, extent=(-3.9, 4.1, -6.1, 3.9), interpolation="bilinear")
        plt.axis("off")
        plt.axis("equal")
        
    def setSession(self, session, filteredCoordinates=None, speed=False, filterSpeed=None):
        if filteredCoordinates is None:
            tracking = session.readTracking(inCm=True)
            coords = 0.5*(tracking.leftEar + tracking.rightEar)
            wallDists = pd.concat((coords.x, coords.y, 49-coords.x, 49-coords.y), axis=1)
            wallDists.columns = ["left", "bottom", "right", "top"]
            closestWallId = wallDists.idxmin(axis=1)
            bodyVec = coords - tracking.tailBase
            bodyDir = np.arctan2(bodyVec.y, bodyVec.x).rename("bodyDirection")
            angleOfWall = closestWallId.replace({'left': np.pi/2, 'top': 0,
                                                 'right': -np.pi/2, 'bottom': np.pi})
            wallAngle = (angleOfWall - bodyDir + 2*np.pi)%(2*np.pi) - np.pi
            minWallDist = wallDists.min(axis=1)
            likelihood = tracking[[("leftEar", "likelihood"),
                                   ("rightEar", "likelihood"),
                                   ("tailBase", "likelihood")]].min(axis=1)
            self.coordinates = np.vstack([np.cos(wallAngle)*minWallDist*30+150,
                                          np.sin(wallAngle)*minWallDist*30+150]).T
            self.coordinates[likelihood.values < 0.9, :] = np.nan
            self.coordinates[likelihood.shift(1).values < 0.9, :] = np.nan
            self.coordinates[minWallDist>4.9, :] = np.nan
        elif not speed:
            coords = filteredCoordinates[["x", "y"]]
            wallDists = pd.concat((coords.x, coords.y, 49-coords.x, 49-coords.y), axis=1)
            wallDists.columns = ["left", "bottom", "right", "top"]
            closestWallId = wallDists.idxmin(axis=1)
            bodyDir = filteredCoordinates.bodyAngle
            angleOfWall = closestWallId.replace({'left': np.pi/2, 'top': 0,
                                                 'right': -np.pi/2, 'bottom': np.pi})
            wallAngle = (angleOfWall - bodyDir + 2*np.pi)%(2*np.pi) - np.pi
            minWallDist = wallDists.min(axis=1)
            self.coordinates = np.vstack([np.cos(wallAngle)*minWallDist*30+150,
                                          np.sin(wallAngle)*minWallDist*30+150]).T
            self.coordinates[minWallDist>4.9, :] = np.nan
        else:
            coords = filteredCoordinates[["x", "y"]]
            wallDists = pd.concat((coords.x, coords.y, 49-coords.x, 49-coords.y), axis=1)
            wallDists.columns = ["left", "bottom", "right", "top"]
            closestWallId = wallDists.idxmin(axis=1)
            bodyDir = np.arctan2(coords.y.diff(), coords.x.diff())
            angleOfWall = closestWallId.replace({'left': np.pi/2, 'top': 0,
                                                 'right': -np.pi/2, 'bottom': np.pi})
            wallAngle = (angleOfWall - bodyDir + 2*np.pi)%(2*np.pi) - np.pi
            minWallDist = wallDists.min(axis=1)
            self.coordinates = np.vstack([np.cos(wallAngle)*minWallDist*30+150,
                                          np.sin(wallAngle)*minWallDist*30+150]).T
            self.coordinates[minWallDist>4.9, :] = np.nan
            if filterSpeed is not None:
                speeds = np.sqrt((coords.diff()**2).sum(axis=1))
                self.coordinates[speeds < filterSpeed]  = np.nan
                
class AllCombinedPlot:
    def __init__(self, block):
        self.schematic = SchematicIntensityPlot(block)
        self.tracking = TrackingIntensityPlot(block)
        taskAreaLimit = block.getWallCorners()[2]-70
        self.bodyDirectionInTask = BodyDirectionPlot(block, positionFilter="x>={}".format(taskAreaLimit))
        self.bodyTurnInTask = BodyTurnPlot(block, positionFilter="x>={}".format(taskAreaLimit))
        self.headTurnInTask = HeadTurnPlot(block, positionFilter="x>={}".format(taskAreaLimit))
        self.gazePointInTask = GazePointPlot(block, positionFilter="x>={}".format(taskAreaLimit))
        self.bodyDirectionOutOfTask = BodyDirectionPlot(block, positionFilter="x<{}".format(taskAreaLimit))
        self.bodyTurnOutOfTask = BodyTurnPlot(block, positionFilter="x<{}".format(taskAreaLimit))
        self.headTurnOutOfTask = HeadTurnPlot(block, positionFilter="x<{}".format(taskAreaLimit))
        self.gazePointOutOfTask = GazePointPlot(block, positionFilter="x<{}".format(taskAreaLimit))
        self.time = TimePlot(block, positionFilter="x<{}".format(taskAreaLimit))
        self.returnBouts = ReturnBoutsPlot(block)
        self.blockActions = BlockActionsIntensityPlot(block)
        
    def draw(self, trace, title, saturation=0.5, traceLabel="Deconvolved activity [z-score]", fig=None):
        if fig is None:
            fig = plt.figure(figsize=(7.5, 9.5))
            
        plt.subplot2grid((24,16), ((0,0)), colspan=8, rowspan=6)
        self.schematic.draw(trace, saturation=saturation)
        plt.subplot2grid((24,16), ((0,8)), colspan=8, rowspan=6)
        self.blockActions.draw(trace, saturation=saturation)
        
        plt.subplot2grid((24,16), ((7,1)), colspan=7, rowspan=3)
        self.returnBouts.draw(trace, saturation=saturation)
        plt.title("Return bouts", fontsize=10, pad=0)
        plt.subplot2grid((24,16), ((6,10)), colspan=6, rowspan=4)
        self.tracking.draw(trace, saturation=saturation)

        plt.subplot2grid((24,16), ((11,1)), colspan=14, rowspan=1)
        plt.gca().xaxis.set_ticks_position('top')
        self.time.draw(trace, saturation=saturation, xlabel="")
        
        plt.subplot2grid((24,16), ((13,0)), colspan=4, rowspan=5)
        self.bodyDirectionInTask.draw(trace, saturation=saturation)
        plt.text(-2,0,"Task area", fontsize=14, rotation="vertical", verticalalignment="center")
        plt.title("Body\ndirection", fontsize=10, pad=0)
        plt.subplot2grid((24,16), ((13,4)), colspan=4, rowspan=5)
        self.bodyTurnInTask.draw(trace, saturation=saturation)
        plt.title("Body\nrotation", fontsize=10, pad=0)
        plt.subplot2grid((24,16), ((13,8)), colspan=2, rowspan=5)
        self.headTurnInTask.draw(trace, saturation=saturation)
        plt.title("Head\ndirection", fontsize=10, pad=0)
        plt.subplot2grid((24,16), ((13,10)), colspan=6, rowspan=5)
        self.gazePointInTask.draw(trace, saturation=saturation)
        plt.title("Gaze\npoint", fontsize=10, pad=0)

        plt.subplot2grid((24,16), ((18,0)), colspan=4, rowspan=5)
        self.bodyDirectionOutOfTask.draw(trace, saturation=saturation)
        plt.text(-2,0,"Other area", fontsize=14, rotation="vertical", verticalalignment="center")
        plt.subplot2grid((24,16), ((18,4)), colspan=4, rowspan=5)
        self.bodyTurnOutOfTask.draw(trace, saturation=saturation)
        plt.subplot2grid((24,16), ((18,8)), colspan=2, rowspan=5)
        self.headTurnOutOfTask.draw(trace, saturation=saturation)
        plt.subplot2grid((24,16), ((18,10)), colspan=6, rowspan=5)
        self.gazePointOutOfTask.draw(trace, saturation=saturation)
        
        plt.suptitle(title)
        plt.tight_layout(w_pad=-1, h_pad=-1, rect=[0.02,0.075,1,0.98])
        cbarAx = fig.add_axes([0.15,0.075,0.7,0.02])
        cbar = plt.colorbar(cax=cbarAx, orientation="horizontal")
        cbar.ax.set_xlabel(traceLabel)
        return fig

class RoiPlot:
    '''
    Used to plot filled ROIs with colored intensities. Useful for showing the spatial distribution of
    factor loadings, correlations, coefficients etc.
    
    The paths for the ROIs are calculated on object creation, and can then be reused for different
    colors by repeatingly calling the draw function.
    '''
    def __init__(self, sess, peakFrac=0.6):
        masks = sess.readROIs()
        self.contours = []
        for _, mask in masks.iteritems():
            mask = mask.unstack()
            peak = mask.values.max()
            contour = skimage.measure.find_contours(mask.values.T, peakFrac*peak)[0]
            self.contours.append(contour)
    
    def draw(self, values, saturation, alpha=0.5, colorbar=True, colorLabel="", positive=False):
        if positive:
            clipped = np.clip(values / saturation, 0, 1)
            colors = plt.cm.viridis(clipped)
        else:
            clipped = np.clip(values / saturation, -1, 1)
            colors = plt.cm.RdYlBu_r(clipped*0.5 + 0.5)
        for contour, color in zip(self.contours, colors):
            plt.fill(contour[:,0], -contour[:,1], color=color, alpha=alpha, edgecolor="k")
        plt.axis("equal")
        plt.axis("off")
        if colorbar:
            cbarAx = plt.gcf().add_axes([0.8,0.15,0.02,0.7])
            if positive:
                cb1 = matplotlib.colorbar.ColorbarBase(cbarAx, cmap=plt.cm.viridis,
                                               norm=matplotlib.colors.Normalize(vmin=0, vmax=saturation),
                                               orientation='vertical',
                                               ticks=[0, saturation/2.0, saturation],
                                               label=colorLabel)
            else:
                cb1 = matplotlib.colorbar.ColorbarBase(cbarAx, cmap=plt.cm.RdYlBu_r,
                                               norm=matplotlib.colors.Normalize(vmin=-saturation, vmax=saturation),
                                               orientation='vertical',
                                               ticks=[-saturation,0,saturation],
                                               label=colorLabel)
class SwitchSchematicPlot(IntensityPlot):
    
    def __init__(self, session=None, saturation=1.0, smoothing=4, linewidth=2, portRadius=1.0):
        self.saturation = saturation
        self.smoothing = smoothing
        self.linewidth = linewidth
        self.portRadius = portRadius
        self.mask = slice(None, None) #No mask, use all values
        self.clearBuffer()
        self.setSession(session)
    
    def clearBuffer(self):
        self.valueCanvas = np.zeros((300, 600), np.float64)
        self.normCanvas = np.zeros((300, 600), np.float64)
    
    def _drawSchema(self, im, alpha, ax=None):
        '''Internal function, do not call directly.'''
        imshowWithAlpha(im, alpha, self.saturation, origin="lower",
                        extent=(-15,15,-7.5,7.5), zorder=-100000)
        plt.xlim(-15, 15)
        plt.ylim(-7.5, 7.5)
        r = self.portRadius
        lw = self.linewidth
        sqrt2 = np.sqrt(2)
        ax = plt.gca()
        #lw=1
        
        ax.add_artist(matplotlib.patches.Arc((8, 0), 6, 6, 0, -90, -45, lw=lw, edgecolor="k"))
        ax.add_artist(matplotlib.patches.Arc((8, 0), 4, 4, 0, -45, 90, lw=lw, edgecolor="k"))
        ax.add_artist(matplotlib.patches.Arc((8, 0), 8, 8, 0, -45, 90, lw=lw, edgecolor="k"))
        ax.plot([8+1/sqrt2, 8+5/sqrt2], [-1/sqrt2, -5/sqrt2], 'k-', lw=lw)
        ax.add_artist(matplotlib.patches.Arc((-8, 0), 6, 6, 0, 90, 135, lw=lw, edgecolor="k"))
        ax.add_artist(matplotlib.patches.Arc((-8, 0), 4, 4, 0, 135, 270, lw=lw, edgecolor="k"))
        ax.add_artist(matplotlib.patches.Arc((-8, 0), 8, 8, 0, 135, 270, lw=lw, edgecolor="k"))
        ax.plot([-8-1/sqrt2, -8-5/sqrt2], [1/sqrt2, 5/sqrt2], 'k-', lw=lw)
        ax.add_artist(matplotlib.patches.Arc((3, 0), 4, 4, 0, 90, 270, lw=lw, edgecolor="k"))
        ax.add_artist(matplotlib.patches.Arc((-3, 0), 4, 4, 0, -90, 90, lw=lw, edgecolor="k"))
        ax.plot([-8, 8], [4, 4], 'k-', lw=lw)
        ax.plot([-8, 8], [-4, -4], 'k-', lw=lw)
        ax.plot([3, 8], [-2, -2], 'k-', lw=lw)
        ax.plot([3, 8], [2, 2], 'k-', lw=lw)
        ax.plot([-3, -8], [-2, -2], 'k-', lw=lw)
        ax.plot([-3, -8], [2, 2], 'k-', lw=lw)

        drawRoundedRect(ax, (8, -6), 5, 12, 1, fill=False, edgecolor="k", lw=lw)
        drawRoundedRect(ax, (-3, -6), 6, 12, 1, fill=False, edgecolor="k", lw=lw)
        drawRoundedRect(ax, (-13, -6), 5, 12, 1, fill=False, edgecolor="k", lw=lw)
        drawArrowHead(ax, (-7.5,4), (-8, 4), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (-7.5,2), (-8, 2), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (7.5,-4), (8, -4), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (7.5,-2), (8, -2), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (3.5, 4), (3, 4), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (3.5, 2), (3, 2), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (-3.5, -4), (-3, -4), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (-3.5, -2), (-3, -2), facecolor="k", edgecolor="k")
       
        wdBase = np.array((-8.7, 1.5))
        drawWaterDrop(ax, wdBase+(-0, 0), 0.4, cross=False, lw=lw)
        drawWaterDrop(ax, wdBase+(-0.7, 0.7), 0.4, cross=True, lw=lw)
        drawWaterDrop(ax, wdBase+(-0.7*2, 0.7*2), 0.4, cross=False, lw=lw)
        drawWaterDrop(ax, wdBase+(-0.7*3, 0.7*3), 0.4, cross=True, lw=lw)

        wdBase = np.array((8.7, -1.3))
        drawWaterDrop(ax, wdBase+(0, 0), 0.4, cross=False, lw=lw)
        drawWaterDrop(ax, wdBase+(0.7, -0.7), 0.4, cross=True, lw=lw)
        drawWaterDrop(ax, wdBase+(0.7*2, -0.7*2), 0.4, cross=False, lw=lw)
        drawWaterDrop(ax, wdBase+(0.7*3, -0.7*3), 0.4, cross=True, lw=lw)
    
        ax.axis("off")
        
    def setSession(self, session):
        lfa = session.labelFrameActions(reward="fullTrial", switch=True, splitCenter=True)
        schematicCoord = fancyVizUtils.switchSchematicCoordinates(lfa)*20
        schematicCoord.x += 300
        schematicCoord.y += 150
        self.coordinates = schematicCoord.values

class OpenFieldSchematicPlot(IntensityPlot):
    
    def __init__(self, session=None, saturation=1.0, smoothing=3, linewidth=2,
                 cmap="RdYlBu_r"):
        self.saturation = saturation
        self.smoothing = smoothing
        self.linewidth = linewidth
        self.cmap = cmap
        self.mask = slice(None, None) #No mask, use all values
        self.clearBuffer()
        if session is not None:
            self.setSession(session)
    
    def clearBuffer(self):
        self.valueCanvas = np.zeros((151, 300), np.float64)
        self.normCanvas = np.zeros((151, 300), np.float64)
    
    def _drawSchema(self, im, alpha, ax=None):
        '''Internal function, do not call directly.'''
        imshowWithAlpha(im, alpha, self.saturation, origin="lower",
                        extent=(-3,3,-1,2), zorder=-100000, cmap=self.cmap)
        if ax is None:
            ax = plt.gca()
        lw = self.linewidth
        ax.add_artist(matplotlib.patches.Arc((1.5, 0), 2, 2, 0, 30, 180, lw=lw, edgecolor="k"))
        ax.add_artist(matplotlib.patches.Arc((-1.5, 0), 2, 2, 0, 0, 150, lw=lw, edgecolor="k"))
        plt.plot([0, -0.5, 0.5, 0, 0], [-0.75, 0, 0, -0.75, 2.0], lw=lw, color="k")
        drawArrowHead(ax, (0, 1.75), (0, 2.0), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (1.5+np.cos(np.pi/4), np.sin(np.pi/4)),
                          (1.5+np.cos(np.pi/6), np.sin(np.pi/6)), facecolor="k", edgecolor="k")
        drawArrowHead(ax, (-1.5-np.cos(np.pi/4), np.sin(np.pi/4)),
                          (-1.5-np.cos(np.pi/6), np.sin(np.pi/6)), facecolor="k", edgecolor="k")
        #plt.axis("square")
        plt.xlim(-3, 3)
        plt.ylim(-1,2.5)
        ax.axis("off")
        
    def setSession(self, session):
        #TODO: This part should probably be stored in the dataframe
        cachedDataPath = "cache/segmentedBehavior.pkl"
        segmentedBehavior = pd.read_pickle(cachedDataPath)
        sb = segmentedBehavior.loc[str(session)].reset_index()
        
        lastFrame = sb.stopFrame.iloc[-1]
        sb = sb[["actionNo", "startFrame", "numFrames", "netTurn", "behavior"]]
        sb["nextBehavior"] = sb.behavior.shift(-1)
        sb = sb.set_index("startFrame").reindex(np.arange(lastFrame), method="ffill")
        sb["progress"] = sb.groupby("actionNo").cumcount() / sb.numFrames
        m = sb.behavior=="leftTurn"
        rad = 0.5 + 0.5*(sb[m].netTurn / 150.0)
        ang = np.deg2rad(sb[m].progress * sb[m].netTurn)
        sb.loc[m, "x"] = -1.5 + rad*np.cos(ang)
        sb.loc[m, "y"] = rad*np.sin(ang)
        m = sb.behavior=="rightTurn"
        rad = 0.5 + 0.5*(-sb[m].netTurn / 150.0)
        ang = np.deg2rad(-180 + sb[m].progress * sb[m].netTurn)
        sb.loc[m, "x"] = 1.5 + rad*np.cos(ang)
        sb.loc[m, "y"] = rad*np.sin(ang)
        m = sb.behavior=="running"
        sb.loc[m, "x"] = 0
        sb.loc[m, "y"] = 2*sb[m].progress
        m = sb.behavior=="stationary"
        sb.loc[m, "y"] = 0.75*(sb[m].progress-1)
        m = np.logical_and(sb.behavior=="stationary", sb.nextBehavior=="running")
        sb.loc[m, "x"] = 0
        m = np.logical_and(sb.behavior=="stationary", sb.nextBehavior=="leftTurn")
        sb.loc[m, "x"] = -0.5 * sb[m].progress
        m = np.logical_and(sb.behavior=="stationary", sb.nextBehavior=="rightTurn")
        sb.loc[m, "x"] = 0.5 * sb[m].progress
        self.rawCoords = sb
        
        schematicCoord = sb[["x", "y"]]*50
        schematicCoord["x"] += 150
        schematicCoord["y"] += 50
        self.coordinates = schematicCoord.values
        
def imshowWithAlpha(im, alpha, saturation, **kwargs):
    im = np.clip(im / saturation, -1, 1)
    if "cmap" not in kwargs: kwargs["cmap"] = "RdYlBu_r"
    cmap = plt.cm.get_cmap(kwargs["cmap"])
    colors = cmap(im*0.5 + 0.5) # -> colors are hard-coded
    colors[:,:,3] = np.clip(alpha, 0, 1)
    kwargs["vmin"] = -saturation # this should only affect the colorbar
    kwargs["vmax"] = saturation
    if "interpolation" not in kwargs: kwargs["interpolation"] = "nearest"
    plt.imshow(colors, **kwargs)

def drawBinnedSchematicPlot(binColors, lw = 2, boxRadius=0.4, saturation=1.0, mWidth=0.75, cmap=plt.cm.RdYlBu_r):
    '''
    Use a shape similar to the SchematicIntensityPlot to show some value per action.
    
    Arguments:
    binColors -- A dictonary or a pandas Series with action labels as keys. For each label,
                 if the value is a number it will be converted to a color using the same color
                 map as in SchematicIntensityPlot. Otherwise it is treated as a matplotlib color.
                 For movement actions (mC2L, mC2R, mL2C, mR2C), it is also allowed to have a list
                 as a value. That action is then split into on subcompartment for each element in the
                 list. The elements in the list can be either values or a matplotlib color.
                
    Example:
    >>> drawBinnedSchematicPlot({'mL2C': [-1, -0.5], 'pC2R': 0, 'pC2L': "gray", 'mC2R': [0.25, 0.5, 1.0]})
    '''
    
    c = dict()
    for action, color in binColors.items():
        if isinstance(color, numbers.Number):
            normed = np.clip(color / saturation, -1, 1)
            c[action] = cmap(normed*0.5 + 0.5)
        elif isinstance(color, (list, np.ndarray, pd.Series)):
            c[action] = []
            for x in color:
                if isinstance(x, numbers.Number):
                    normed = np.clip(x / saturation, -1, 1)
                    x = cmap(normed*0.5 + 0.5)
                c[action].append(x)
        else:
            c[action] = color
    for action, color in c.items():
        if action[0] == "m" and not isinstance(color, list):
            c[action] = [color]
            
    r = boxRadius
    
    if "pC" in c:
        drawRoundedRect(plt.gca(), (-0.75, -1), 1.5, 2, [r, r, r, r], fill=True, lw=lw, facecolor=c["pC"], edgecolor="k")
    if "pC2R" in c:
        drawRoundedRect(plt.gca(), ( 0.05, -1), 0.7, 2, [0, 0, r, r], fill=True, lw=lw, facecolor=c["pC2R"], edgecolor="k")
    if "pC2L" in c:
        drawRoundedRect(plt.gca(), (-0.75, -1), 0.7, 2, [r, r, 0, 0], fill=True, lw=lw, facecolor=c["pC2L"], edgecolor="k")

    if "pL2Co" in c:
        drawRoundedRect(plt.gca(), (-4.75, -1),  0.7, 1.2, [r, 0, 0, 0], fill=True, lw=lw, facecolor=c["pL2Co"], edgecolor="k")
    if "pL2Cr" in c:
        drawRoundedRect(plt.gca(), (-3.95, -1),  0.7, 1.2, [0, 0, 0, r], fill=True, lw=lw, facecolor=c["pL2Cr"], edgecolor="k")
    if "dL2C" in c:
        drawRoundedRect(plt.gca(), (-4.75, 0.3), 1.5, 0.7, [0, r, r, 0], fill=True, lw=lw, facecolor=c["dL2C"], edgecolor="k")
    if "pL2C" in c:
        drawRoundedRect(plt.gca(), (-4.75, -1),  1.5, 1.2, [r, 0, 0, r], fill=True, lw=lw, facecolor=c["pL2C"], edgecolor="k")

    if "pR2Cr" in c:
        drawRoundedRect(plt.gca(), (3.25, -1), 0.7, 1.2,  [r, 0, 0, 0], fill=True, lw=lw, facecolor=c["pR2Cr"], edgecolor="k")
    if "pR2Co" in c:
        drawRoundedRect(plt.gca(), (4.05, -1), 0.7, 1.2,  [0, 0, 0, r], fill=True, lw=lw, facecolor=c["pR2Co"], edgecolor="k")
    if "dR2C" in c:
        drawRoundedRect(plt.gca(), (3.25, 0.3), 1.5, 0.7, [0, r, r, 0], fill=True, lw=lw, facecolor=c["dR2C"], edgecolor="k")
    if "pR2C" in c:
        drawRoundedRect(plt.gca(), (3.25, -1),  1.5, 1.2, [r, 0, 0, r], fill=True, lw=lw, facecolor=c["pR2C"], edgecolor="k")

    xx = np.linspace(-1,1)
    yy = 1-xx*xx
    normal_x = 2.0*(xx[::-1])
    normal_y = 4.0
    normal_len  = np.sqrt(normal_x*normal_x + normal_y*normal_y)
    normal_x *= mWidth / 2 / normal_len
    normal_y *= mWidth / 2 / normal_len

    if "mL2C" in c:
        drawSegments(xx*2 - 2, -yy - 1, normal_x, normal_y, c["mL2C"],
                     (-5, 0), (-3, -1), edgecolor="k", lw=lw)
    if "mR2C" in c:
        drawSegments(xx[::-1]*2 + 2, -yy[::-1] - 1, normal_x[::-1], normal_y[::-1], c["mR2C"], 
                     (0, 5), (-3, -1), edgecolor="k", lw=lw)
    normal_x = 1.8*(xx[::-1])
    normal_y = -4.0
    normal_len  = np.sqrt(normal_x*normal_x + normal_y*normal_y)
    normal_x *= mWidth / 2 / normal_len
    normal_y *= mWidth / 2 / normal_len
    if "mC2L" in c:
        drawSegments(xx[::-1]*1.8 - 2.2, yy[::-1] + 1, normal_x[::-1], normal_y[::-1], c["mC2L"],
                     (-5, 0), (1, 3), edgecolor="k", lw=lw)
    if "mC2R" in c:
        drawSegments(xx*1.8 + 2.2, yy + 1, normal_x, normal_y, c["mC2R"],
                     (0, 5), (1, 3), edgecolor="k", lw=lw)
    if not 'pL2C' in c:
        drawWaterDrop(plt.gca(), np.array([-2.75, -0.5]), 0.3)
        drawWaterDrop(plt.gca(), np.array([2.75, -0.5]), 0.3)
        drawWaterDrop(plt.gca(), np.array([-4.6, -1.5]), 0.3, True)
        drawWaterDrop(plt.gca(), np.array([4.6, -1.5]), 0.3, True)
    plt.axis("square")
    plt.xlim(-5,5)
    plt.ylim(-2.5,2.5)
    plt.axis("off")

def drawArcArrow(rad, start, stop, lw=0.5):
    phi = np.linspace(start,stop,100)
    xx = np.cos(phi)*rad
    yy = np.sin(phi)*rad
    plt.plot(xx, yy, 'k', lw=lw)
    drawArrowHead(plt.gca(), (xx[-12], yy[-12]), (xx[-1], yy[-1]), facecolor="k", edgecolor="k")

def drawWaterDrop(ax, coords, size, cross=False, facecolor='skyblue', alpha=1.0, lw=0.75, **kwargs):
    vertices = np.array([(-0.1,1.0), (-0.15,0.15), (-0.5,-0.2),
                         (-0.75,-0.5), (-0.75,-1), (0,-1),
                         (0.75,-1), (1, -1), (-0.1,1.0)])
    codes = [matplotlib.path.Path.MOVETO,
             matplotlib.path.Path.CURVE3,
             matplotlib.path.Path.CURVE3]+[matplotlib.path.Path.CURVE4]*6
    path = matplotlib.path.Path(vertices*size + coords[np.newaxis, :], codes)
    patch = matplotlib.patches.PathPatch(path, facecolor=facecolor, alpha=alpha, lw=0, transform=ax.transData, **kwargs)
    ax.add_patch(patch)
    if cross:
        ax.plot(coords[0]+size*np.array([-0.5,0.5]), coords[1]+size*np.array([-1,0.4]), c="red", lw=lw)
        ax.plot(coords[0]+size*np.array([-0.5,0.5]), coords[1]+size*np.array([0.4,-1]), c="red", lw=lw)
    return patch

def drawRoundedRect(ax, position, width, height, radius, **kwargs):
    h = height
    w = width
    pos = np.array(position)
    if not isinstance(radius, (list, tuple, np.ndarray)):
        r = radius*np.ones(4)
    else:
        r = radius
    Path = matplotlib.path.Path
    path_data = [
        (Path.MOVETO, [0, r[0]]),
        (Path.LINETO, [0, h-r[1]]),
        (Path.CURVE3, [0, h]),
        (Path.CURVE3, [r[1], h]),
        (Path.LINETO, [w-r[2], h]),
        (Path.CURVE3, [w, h]),
        (Path.CURVE3, [w, h-r[2]]),
        (Path.LINETO, [w, r[3]]),
        (Path.CURVE3, [w, 0]),
        (Path.CURVE3, [w-r[3], 0]),
        (Path.LINETO, [r[0], 0]),
        (Path.CURVE3, [0, 0]),
        (Path.CURVE3, [0, r[0]]),
        (Path.CLOSEPOLY, [0, 0])]
    codes, verts = zip(*path_data)
    verts = np.array(verts) + pos[np.newaxis, :]
    path = Path(verts, codes)
    patch = matplotlib.patches.PathPatch(path, **kwargs)
    return ax.add_artist(patch)

def drawArrowHead(ax, base, tip, aspect=0.4, **kwargs):
    base = np.array(base)
    tip = np.array(tip)
    d = tip - base
    n = aspect*np.array([-d[1], d[0]])
    corners = np.array([base + n, tip, base-n])
    return ax.add_patch(plt.Polygon(corners, **kwargs))

def drawSegments(x, y, nx, ny, colors, xlim=(-6, 6), ylim=(-3, 3), **kwargs):
    segSize = len(x)/len(colors)
    for i in range(len(colors)):
        sl = slice(int(segSize * i), int(segSize * (i+1)))
        polyX = np.concatenate((x[sl] - nx[sl], x[sl][::-1] + nx[sl][::-1]))
        polyY = np.concatenate((y[sl] - ny[sl], y[sl][::-1] + ny[sl][::-1]))
        polyX = np.clip(polyX, *xlim)
        polyY = np.clip(polyY, *ylim)
        plt.gca().add_patch(plt.Polygon(np.vstack((polyX, polyY)).T, fill=True, facecolor=colors[i], **kwargs))