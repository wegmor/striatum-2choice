import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyximport; pyximport.install()
import scipy.ndimage
import matplotlib.backends.backend_pdf

from . import fancyVizUtils, trackingGeometryUtils


def imshowWithAlpha(im, alpha, saturation, **kwargs):
    density = np.clip(im, saturation, -1, 1)
    colors = plt.cm.RdYlBu_r(de*0.5 + 0.5)
    colors[:,:,3] = np.clip(alpha, 0, 1)
    kwargs["vmin"] = -saturation
    kwargs["vmax"] = saturation
    if "cmap" not in kwargs: kwargs["cmap"] = "RdYlBu_r"
    if "interpolation" not in kwargs: kwargs["interpolation"] = "nearest"
    plt.imshow(colors, **kwargs)

def drawTaskSchematic(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization, saturation,
                    origin="lower", extent=(-5,5,-2.5,2.5), zorder=-100000)
    
    plt.xlim(-5,5)
    plt.ylim(-2.75,2.5)
    elLeft = matplotlib.patches.Ellipse((-4,0),1.5,2.0,edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    elCenter = matplotlib.patches.Ellipse((0,0),1.5,2.0,edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    elRight = matplotlib.patches.Ellipse((4,0),1.5,2.0,edgecolor="k", facecolor="none", lw=2, zorder=-10000)
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
    
def drawTracking(density, normalization, saturation=0.5, background=None):
    if isinstance(background, str):
        background = PIL.Image.open(background)
    elif background is None:
        background = PIL.Image.open("/home/emil/2choice/boxBackground.png")
    plt.imshow(background, alpha=0.5)
    imshowWithAlpha(density / normalization, normalization, saturation)
    plt.axis("off")

def drawBodyDirection(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization,
                    saturation, extent=(-1.5,1.5,-1.5,1.5))
    plt.axis("off")
    plt.axis("equal")

def drawBodyTurn(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization,
                    saturation, extent=(-1.5,1.5,-1.5,1.5))
    arcArrow(1, -0.1, -2)
    arcArrow(1 , 0.1, 2)
    plt.axis("off")
    plt.axis("equal")

def drawGazePoint(density, normalization, saturation=0.5):
    imshowWithAlpha(density / normalization, normalization, saturation, extent=(-1.5,1.5,-1.5,1.5))
    plt.axis("off")
    plt.axis("equal")
    
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


def defaultCanvasSize():
    return {"schematic": (251, 501),
            "tracking":  (304, 400),
            "headDir":   (301, 151),
            "bodyDir":   (301, 301),
            "bodyTurn":  (301, 301),
            "gazePoint": (304, 400)}