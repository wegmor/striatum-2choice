import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def findHighs(series):
    start = 0
    high = False
    highs = []
    for i, v in zip(series.index, series):
        if not high and v==1:
            high = True
            start = i
        elif high and v == 0:
            high = False
            highs.append((start, i))
    if high:
        highs.append((start, series.index[-1]))
    return pd.DataFrame(highs, columns=["start", "stop"])
    
def plotPortEntries(beams, **kwargs):
    portEntries = findHighs(beams)
    for pe in portEntries.itertuples():
        plt.axvspan(pe.start/20.0, pe.stop/20.0, **kwargs)

def plotSingleNeuron(trace, sensorValues, offset=0, timePerRow=60, nRows=15, title=None, filename=None, heightPerRow=0.3):
    sensorValues = sensorValues.set_index("frameNo").sort_index()
    rewardTimes, = np.nonzero(sensorValues.rewardNo.diff().dropna())
    fig = plt.figure(figsize=(7.5,2.5+nRows*heightPerRow))
    for i in range(nRows):
        plt.subplot(nRows,1,i+1)
        sl = slice(offset*20 + i*20*timePerRow, offset*20 + (i+1)*20*timePerRow)
        plotPortEntries(sensorValues.loc[sl].beamL, color="C1", alpha=.4)
        plotPortEntries(sensorValues.loc[sl].beamC, color="C4", alpha=.4)
        plotPortEntries(sensorValues.loc[sl].beamR, color="C2", alpha=.4)
        for rt in rewardTimes:
            if rt > offset*20 + (i+1)*20*timePerRow: break
            plt.axvline(rt/20.0, color="C0")
        plt.plot(trace.iloc[sl], color="k", lw=1, alpha=0.7)
        plt.xlim(offset + timePerRow*i,offset + timePerRow*(i+1))
        plt.ylim(-2,12)
        plt.yticks([])
    plt.xlabel("Time [seconds]")
    sns.despine()
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout(rect=(0,0.085,1,0.95),h_pad=0.2)
    legendAx = fig.add_axes((0, 0.05, 1, 0.05), frameon=False)
    handles = [
        matplotlib.lines.Line2D([0], [0], color='k', label='Calcium trace'),
        matplotlib.patches.Patch(facecolor='C1', edgecolor='None', label='Left port', alpha=0.4),
        matplotlib.patches.Patch(facecolor='C4', edgecolor='None', label='Center port', alpha=0.4),
        matplotlib.patches.Patch(facecolor='C2', edgecolor='None', label='Right port', alpha=0.4),
        matplotlib.lines.Line2D([0], [0], color='C0', label='Reward')
    ]
    legendAx.legend(handles=handles, ncol=5, loc="center")
    legendAx.set_xticks([])
    legendAx.set_yticks([])
    sns.despine(ax=legendAx, left=True, bottom=True)
    
def plotMultipleNeurons(traces, sensorValues, offset=0, timePerRow=60, nRows=15, title=None, filename=None,
                        heightPerRow=0.3, yOffset=10):
    sensorValues = sensorValues.set_index("frameNo").sort_index()
    rewardTimes, = np.nonzero(sensorValues.rewardNo.diff().dropna())
    fig = plt.figure(figsize=(7.5,2.5+nRows*heightPerRow))
    for i in range(nRows):
        plt.subplot(nRows,1,i+1)
        sl = slice(offset*20 + i*20*timePerRow, offset*20 + (i+1)*20*timePerRow)
        plotPortEntries(sensorValues.loc[sl].beamL, color="C1", alpha=.4)
        plotPortEntries(sensorValues.loc[sl].beamC, color="C4", alpha=.4)
        plotPortEntries(sensorValues.loc[sl].beamR, color="C2", alpha=.4)
        for rt in rewardTimes:
            if rt > offset*20 + (i+1)*20*timePerRow: break
            plt.axvline(rt/20.0, color="C0")
        y = 0
        for c in traces.columns:
            plt.plot(y+traces[c].iloc[sl], color="k", lw=1, alpha=0.7)
            y += yOffset
        plt.xlim(offset + timePerRow*i,offset + timePerRow*(i+1))
        plt.ylim(-yOffset*0.5, yOffset*traces.shape[1])
        plt.yticks(np.arange(traces.shape[1])*yOffset, traces.columns)
    plt.xlabel("Time [seconds]")
    sns.despine()
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout(rect=(0,0.085,1,0.95),h_pad=0.2)
    legendAx = fig.add_axes((0, 0.05, 1, 0.05), frameon=False)
    handles = [
        matplotlib.lines.Line2D([0], [0], color='k', label='Calcium trace'),
        matplotlib.patches.Patch(facecolor='C1', edgecolor='None', label='Left port', alpha=0.4),
        matplotlib.patches.Patch(facecolor='C4', edgecolor='None', label='Center port', alpha=0.4),
        matplotlib.patches.Patch(facecolor='C2', edgecolor='None', label='Right port', alpha=0.4),
        matplotlib.lines.Line2D([0], [0], color='C0', label='Reward')
    ]
    legendAx.legend(handles=handles, ncol=5, loc="center")
    legendAx.set_xticks([])
    legendAx.set_yticks([])
    sns.despine(ax=legendAx, left=True, bottom=True)