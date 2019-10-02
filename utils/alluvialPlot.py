import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import Rectangle

def singleAlluvium(leftSide, rightSide, color, alpha=0.4, tilt=0.5, xLeft=0, xRight=1):
    pathData = [
        (Path.MOVETO, [xLeft, leftSide[0]]),
        (Path.CURVE4, [xLeft+tilt, leftSide[0]]),
        (Path.CURVE4, [xRight-tilt, rightSide[0]]),
        (Path.CURVE4, [xRight, rightSide[0]]),
        (Path.LINETO, [xRight, rightSide[1]]),
        (Path.CURVE4, [xRight-tilt, rightSide[1]]),
        (Path.CURVE4, [xLeft+tilt, leftSide[1]]),
        (Path.CURVE4, [xLeft, leftSide[1]]),
        (Path.CLOSEPOLY, [xLeft, leftSide[0]])
    ]
    codes, verts = zip(*pathData)
    verts = np.array(verts)# + pos[np.newaxis, :]
    path = Path(verts, codes)
    patch = matplotlib.patches.PathPatch(path, linewidth=0, color=color, alpha=alpha)
    return patch

def alluvialPlot(data, leftSide, rightSide, colormap, ax, alpha=0.4, tilt=0.5, colorByRight=False):
    '''
    Show co-occurencies of datapoints in a dataframe as an alluvial plot.
    
    Arguments:
    data - A pandas DataFrame with one row per observation.
    leftSide - A string indicating which column should be the left side
    rightSide - A string indicating which column should be the right side
    colormap - A dictionary indicating which color to use for each label
    ax - The matplotlib axis to add the alluvial plot to
    alpha - Opacity of the plot
    tilt - Control parameter of the shape (0 to 1). Small values mean
           straighter lines, large values mean sharper turns.
    colorByRight - Color the alluvial according to the right side instead
                   of the left side (defualt False, i.e. use left side)
    '''
    #Calculate y-coordinates (lower and higher) for left and right side respectively
    leftY = data.groupby([leftSide, rightSide]).size()
    leftY = np.cumsum(leftY) / len(data) * 0.9
    leftY = pd.DataFrame({'lower': leftY.shift(1, fill_value=0), 'higher': leftY})
    leftY += (leftY.groupby(level=0).ngroup() / len(leftY.groupby(level=0)) * 0.1)[:, np.newaxis]
    rightY = data.groupby([rightSide, leftSide]).size()
    rightY = np.cumsum(rightY)  / len(data) * 0.9
    rightY = pd.DataFrame({'lower': rightY.shift(1, fill_value=0), 'higher': rightY})
    rightY += (rightY.groupby(level=0).ngroup() / len(rightY.groupby(level=0)) * 0.1)[:, np.newaxis]
    rightY = rightY.reorder_levels([1,0]).reindex_like(leftY)
    
    #Add indiviual alluvia
    for (ind, ls), (ind, rs) in zip(leftY.iterrows(), rightY.iterrows()):
        if ls[1] > ls[0]: #At least one sample
            patch = singleAlluvium(ls, rs, colormap[ind[colorByRight]], alpha, tilt)
            ax.add_artist(patch)

            ax.add_artist(Rectangle((-0.2, ls[0]), 0.2, ls[1]-ls[0], color=colormap[ind[0]],
                          fill=True, lw=0, alpha=alpha))
            ax.add_artist(Rectangle((1, rs[0]), 0.2, rs[1]-rs[0], color=colormap[ind[1]],
                          fill=True, lw=0, alpha=alpha))
