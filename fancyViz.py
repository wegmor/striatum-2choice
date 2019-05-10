import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
import scipy.ndimage
import skimage.measure
import matplotlib.backends.backend_pdf
import PIL
from deprecated import deprecated

from . import fancyVizUtils, trackingGeometryUtils

class IntensityPlot:

    def setMask(self, mask):
        self.mask = mask
    
    def draw(self, trace, ax=None):
        self.clearBuffer()
        self.addTraceToBuffer(trace)
        self.drawBuffer(ax)
    
    def addTraceToBuffer(self, trace):
        trace = np.array(trace).astype(np.float)
        if len(trace) != len(self.coordinates):
            raise ValueError("The trace contains {} frames but the coordinates {}".format(len(trace), len(self.coordinates)))
        fancyVizUtils.integerHistogram(self.coordinates[self.mask], trace[self.mask],
                                       self.valueCanvas, self.normCanvas)
    
    def drawBuffer(self, ax=None,  **kwargs):
        if ax is not None: plt.sca(ax)
        values = scipy.ndimage.gaussian_filter(self.valueCanvas, self.smoothing)
        norm   = scipy.ndimage.gaussian_filter(self.normCanvas, self.smoothing)
        normalized = np.divide(values, norm, out=np.zeros_like(values), where=norm!=0)
        self._drawSchema(normalized, norm*10, **kwargs)
        self.clearBuffer()
        
class SchematicIntensityPlot(IntensityPlot):
    
    def __init__(self, session=None, saturation=1.0, smoothing=4, linewidth=2, waterDrop=True, portRadius=0.4):
        self.saturation = saturation
        self.smoothing = smoothing
        self.linewidth = linewidth
        self.waterDrop = waterDrop
        self.portRadius = portRadius
        self.mask = slice(None, None) #No mask, use all values
        self.clearBuffer()
        self.setSession(session)
    
    def clearBuffer(self):
        self.valueCanvas = np.zeros((251,501), np.float64)
        self.normCanvas = np.zeros((251,501), np.float64)
    
    def _drawSchema(self, im, alpha, ax=None):
        '''Internal function, do not call directly.'''
        imshowWithAlpha(im, alpha, self.saturation, origin="lower",
                        extent=(-5,5,-2.5,2.5), zorder=-100000)
        plt.xlim(-5, 5)
        plt.ylim(-2.75, 2.5)
        r = self.portRadius
        lw = self.linewidth
        
        drawRoundedRect(plt.gca(), ( 0.05, -1), 0.7, 2, [0, 0, r, r], fill=False, lw=lw, zorder=-10000)
        drawRoundedRect(plt.gca(), (-0.75, -1), 0.7, 2, [r, r, 0, 0], fill=False, lw=lw, zorder=-10000)

        drawRoundedRect(plt.gca(), (-4.75, -1),  0.7, 1.2, [r, 0, 0, 0], fill=False, lw=lw, zorder=-10000)
        drawRoundedRect(plt.gca(), (-3.95, -1),  0.7, 1.2, [0, 0, 0, r], fill=False, lw=lw, zorder=-10000)
        drawRoundedRect(plt.gca(), (-4.75, 0.3), 1.5, 0.7, [0, r, r, 0], fill=False, lw=lw, zorder=-10000)

        drawRoundedRect(plt.gca(), (3.25, -1), 0.7, 1.2,  [r, 0, 0, 0], fill=False, lw=lw, zorder=-10000)
        drawRoundedRect(plt.gca(), (4.05, -1), 0.7, 1.2,  [0, 0, 0, r], fill=False, lw=lw, zorder=-10000)
        drawRoundedRect(plt.gca(), (3.25, 0.3), 1.5, 0.7, [0, r, r, 0], fill=False, lw=lw, zorder=-10000)
        
        xx = np.linspace(-1,1)
        yy = 1-xx*xx
        plt.plot(xx*1.8+2.2, yy+1,'k', lw=lw, zorder=-10000)
        plt.plot(xx*-1.8-2.2,yy+1,'k', lw=lw, zorder=-10000)
        plt.plot(xx*2+2,-yy-1,'k', lw=lw, zorder=-10000)
        plt.plot(xx*-2-2,-yy-1,'k', lw=lw, zorder=-10000)
        plt.arrow(3.9,1.1,0.1,-0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
        plt.arrow(-3.9,1.1,-0.1,-0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
        plt.arrow(-0.1,-1.1,0.1,0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
        plt.arrow(0.1,-1.1,-0.1,0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
        if self.waterDrop:
            drawWaterDrop(plt.gca(), np.array([-2.75, -0.5]), 0.3, lw=lw)
            drawWaterDrop(plt.gca(), np.array([2.75, -0.5]), 0.3, lw=lw)
            drawWaterDrop(plt.gca(), np.array([-4.6, -1.5]), 0.3, True, lw=lw)
            drawWaterDrop(plt.gca(), np.array([4.6, -1.5]), 0.3, True, lw=lw)
        plt.axis("off")
        
    def setSession(self, session):
        lfa = session.labelFrameActions(reward=True, switch=False)
        schematicCoord = fancyVizUtils.taskSchematicCoordinatesFrameLabels(lfa)*50
        schematicCoord.x += 250
        schematicCoord.y += 125
        self.coordinates = schematicCoord.values
        
class TrackingIntensityPlot(IntensityPlot):
    def __init__(self, block=None, smoothing=5, background="/home/emil/2choice/boxBackground.png"):
        self.smoothing = smoothing
        self.canvasSize = (304, 400)
        
        if block is not None and block.meta.cohort=="2019":
            self.canvasSize = (750, 872)
            background = "/home/emil/2choice/boxBackgroundHighRes.png"
            
        if isinstance(background, str):
            self.background = PIL.Image.open(background)
        elif isinstance(background, PIL.Image):
            self.background = background
        else:
            raise ValueError("Unknown background format")
            
        if block is not None:
            self.setDefaultCoordinates(block)
            
    def draw(self, trace, saturation=0.5 ,ax=None):
        IntensityPlot.draw(self, trace, saturation, ax)
        plt.imshow(self.background, alpha=0.5, zorder=-1000)
        plt.axis("off")
        
    def setDefaultCoordinates(self, block):
        tracking = block.readTracking()
        headCoordinates = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
        likelihood = tracking[[("leftEar", "likelihood"),
                           ("rightEar", "likelihood"),
                           ("tailBase", "likelihood")]].min(axis=1)
        self.setCoordinates(headCoordinates.values, (likelihood.values>0.9))
        
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
    def __init__(self, block=None, smoothing=6, positionFilter=None):
        self.smoothing = smoothing
        self.canvasSize = (301, 301)
        if block is not None:
            self.setDefaultCoordinates(block, positionFilter)
            
    def draw(self, trace, saturation=0.5, ax=None):
        IntensityPlot.draw(self, trace, saturation, ax, extent=(-1.5,1.5,-1.5,1.5))
        drawArcArrow(1, -0.1, -2)
        drawArcArrow(1 , 0.1, 2)
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
        turningSpeed = trackingGeometryUtils.angleDiff(bodyDir.shift(1), bodyDir)
        mask = np.logical_and(mask, np.abs(turningSpeed)<np.pi/6)
        bodyTurnCoord = np.vstack([np.cos(turningSpeed*6)*100+150,
                                   np.sin(turningSpeed*6)*100+150]).T
        self.setCoordinates(bodyTurnCoord, mask)
        
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
    def __init__(self, block, peakFrac=0.6):
        masks = block.readROIs()
        self.contours = []
        for _, mask in masks.iteritems():
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

@deprecated(reason="Use flag in SchematicIntensityPlot instead")
class SchematicIntensityPlotForFrameLabels(SchematicIntensityPlot):
    def setDefaultCoordinates(self, block, maxLen=20):
        apf = block.labelFrameActions()
        schematicCoord = fancyVizUtils.taskSchematicCoordinatesFrameLabels(apf.reset_index())*50
        schematicCoord.x += 250
        schematicCoord.y += 125
        #mask = np.ones(len(schematicCoord), np.bool_)
        #A temporary hack to avoid NaN in the traces (should rewrite actual code to correct for NaNs)
        mask = np.logical_not(block.readDeconvolvedTraces().isna().any(axis=1).values)
        self.setCoordinates(schematicCoord.values, mask)        
            
def imshowWithAlpha(im, alpha, saturation, **kwargs):
    im = np.clip(im / saturation, -1, 1)
    colors = plt.cm.RdYlBu_r(im*0.5 + 0.5)
    colors[:,:,3] = np.clip(alpha, 0, 1)
    kwargs["vmin"] = -saturation
    kwargs["vmax"] = saturation
    if "cmap" not in kwargs: kwargs["cmap"] = "RdYlBu_r"
    if "interpolation" not in kwargs: kwargs["interpolation"] = "nearest"
    plt.imshow(colors, **kwargs)

    
@deprecated(reason="Please use the object-oriented interface instead.")
def drawTaskSchematic(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization, saturation,
                    origin="lower", extent=(-5,5,-2.5,2.5), zorder=-100000)
    
    plt.xlim(-5,5)
    plt.ylim(-2.75,2.5)
    elLeft = matplotlib.patches.Ellipse((-4,0),1.5,2.0, edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    elCenter = matplotlib.patches.Ellipse((0,0),1.5,2.0, edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    elRight = matplotlib.patches.Ellipse((4,0),1.5,2.0, edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    plt.gca().add_artist(elLeft)
    plt.gca().add_artist(elCenter)
    plt.gca().add_artist(elRight)
    xx = np.linspace(-1,1)
    yy = 1-xx*xx
    plt.plot(xx*2+2,yy+1,'k', lw=2, zorder=-10000)
    plt.plot(xx*-2-2,yy+1,'k', lw=2, zorder=-10000)
    plt.plot(xx*2+2,-yy-1,'k', lw=2, zorder=-10000)
    plt.plot(xx*-2-2,-yy-1,'k', lw=2, zorder=-10000)
    plt.arrow(3.9,1.1,0.1,-0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.arrow(-3.9,1.1,-0.1,-0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.arrow(-0.1,-1.1,0.1,0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.arrow(0.1,-1.1,-0.1,0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.axis("off")

def drawArcArrow(rad, start, stop):
    phi = np.linspace(start,stop,100)
    plt.plot(np.cos(phi)*rad, np.sin(phi)*rad, 'k', lw=0.5)
    if start<stop:
        plt.arrow(np.cos(phi[-1])*rad, np.sin(phi[-1])*rad, -np.sin(phi[-1])*0.1, np.cos(phi[-1])*0.1,
              head_width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    else:
        plt.arrow(np.cos(phi[-1])*rad, np.sin(phi[-1])*rad, np.sin(phi[-1])*0.1, -np.cos(phi[-1])*0.1,
              head_width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
        
@deprecated(reason="Please use the object-oriented interface instead.")        
def drawHeadDirection(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization, saturation, extent=(0,1.5,-1.5,1.5))
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

@deprecated(reason="Please use the object-oriented interface instead.")
def drawTracking(density, normalization, saturation=0.5, background=None):
    if isinstance(background, str):
        background = PIL.Image.open(background)
    elif background is None:
        background = PIL.Image.open("/home/emil/2choice/boxBackground.png")
    plt.imshow(background, alpha=0.5)
    imshowWithAlpha(density / normalization, normalization, saturation)
    plt.axis("off")
    
@deprecated(reason="Please use the object-oriented interface instead.")
def drawBodyDirection(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization,
                    saturation, extent=(-1.5,1.5,-1.5,1.5))
    plt.axis("off")
    plt.axis("equal")
    
@deprecated(reason="Please use the object-oriented interface instead.")
def drawBodyTurn(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization,
                    saturation, extent=(-1.5,1.5,-1.5,1.5))
    arcArrow(1, -0.1, -2)
    arcArrow(1 , 0.1, 2)
    plt.axis("off")
    plt.axis("equal")
    
@deprecated(reason="Please use the object-oriented interface instead.")
def drawGazePoint(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization, saturation, extent=(-1.5,1.5,-1.5,1.5))
    plt.axis("off")
    plt.axis("equal")
    
@deprecated(reason="Please use the object-oriented interface instead.")    
def calculateAllPlotCoordinates(block):
    '''
    Calculate the coordinates in all plot types for each frame.
    
    Arguments:
    block -- A 2-choice block
    
    Returns:
    A pandas Dataframe where each row containes the coordinates in each plot type
    for the respective video frame.
    '''
    tracking = block.readTracking()
    apf = block.calcActionsPerFrame()
    headDir    = trackingGeometryUtils.calcHeadDirections(tracking)
    bodyDir    = trackingGeometryUtils.calcBodyDirection(tracking)
    turningSpeed = trackingGeometryUtils.angleDiff(bodyDir.shift(1), bodyDir)
    projSpeed  = trackingGeometryUtils.calcProjectedSpeed(tracking)
    gazePoint  = trackingGeometryUtils.calcGazePoint(tracking)
    likelihood = tracking[[("leftEar", "likelihood"),
                           ("rightEar", "likelihood"),
                           ("tailBase", "likelihood")]].min(axis=1)
    schematicCoord = fancyVizUtils.taskSchematicCoordinates(apf.reset_index())*50
    schematicCoord.x += 250
    schematicCoord.y += 125
    trackingCoord = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
    headDirRadius = np.clip(1 + np.sign(headDir)*headDir.diff()*0.5, 0.5, 1.5)
    headDirCoord = pd.DataFrame({'x': np.cos(headDir)*100*headDirRadius,
                                 'y': np.sin(headDir)*100*headDirRadius+150}, index=apf.index)
    bodyDirRadius = np.clip(1 + projSpeed / 50.0, 0.5, 1.5)
    bodyDirCoord = pd.DataFrame({'x': np.cos(bodyDir)*100*bodyDirRadius+150,
                                 'y': np.sin(bodyDir)*100*bodyDirRadius+150}, index=apf.index)
    bodyTurnCoord = pd.DataFrame({'x': np.cos(turningSpeed)*100+150,
                                  'y': np.sin(turningSpeed)*100+150}, index=apf.index)
    combined = pd.concat([schematicCoord, trackingCoord, headDirCoord, bodyDirCoord, bodyTurnCoord, gazePoint], axis=1,
                          keys=("schematic", "tracking", "headDir", "bodyDir", "bodyTurn", "gazePoint"))
    combined["likelihood"] = likelihood
    return combined

    
@deprecated(reason="Please use the object-oriented interface instead.")
def defaultCanvasSize():
    return {"schematic": (251, 501),
            "tracking":  (304, 400),
            "headDir":   (301, 151),
            "bodyDir":   (301, 301),
            "bodyTurn":  (301, 301),
            "gazePoint": (304, 400)}

def drawWaterDrop(ax, coords, size, cross=False, facecolor='skyblue', alpha=1.0, lw=0.75):
    vertices = np.array([(-0.1,1.0), (-0.15,0.15), (-0.5,-0.2),
                         (-0.75,-0.5), (-0.75,-1), (0,-1),
                         (0.75,-1), (1, -1), (-0.1,1.0)])
    codes = [matplotlib.path.Path.MOVETO,
             matplotlib.path.Path.CURVE3,
             matplotlib.path.Path.CURVE3]+[matplotlib.path.Path.CURVE4]*6
    path = matplotlib.path.Path(vertices*size + coords[np.newaxis, :], codes)
    patch = matplotlib.patches.PathPatch(path, facecolor=facecolor, alpha=alpha, lw=0, transform=ax.transData)
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