import pandas as pd
import numpy as np

import pyximport
pyximport.install()
from . import findTrials

class Session:
    
    def __init__(self, hdfFile, meta):
        self.hdfFile = hdfFile
        self.meta = meta
        
    def __str__(self):
        return self.meta.genotype + "_" + self.meta.animal + "_"+self.meta.date

    def readROIs(self):
        '''Read the regions-of-interest (ROIs) for each neuron in this session. Contains the shape
        of the neurons.

        Returns:
        A Pandas panel with all the ROIs stacked on the item axis        
        '''
        path = "/rois/{}/{}/{}".format(genotype, animal, date)
        return pd.read_hdf(self.hdfFile, path)
    
    def readTraces(self, kind="caTraces", fillDropped=True):
        recordings = []
        for rec in self.meta.caRecordings:
            recData = pd.read_hdf(self.hdfFile, "/caRecordings/{}/{}".format(rec, kind))
            if fillDropped:
                completeIndex = np.arange(recData.index.min(), recData.index.max()+0.025, 0.05)
                recData = recData.reindex(completeIndex, method="nearest", tolerance=1e-3)
            recordings.append(recData)
        return pd.concat(recordings)
    
    def readCaTraces(self, fillDropped=True):
        '''Read all calcium traces from this session.

        Arguments:
        fillDropped -- If True, use time index of the recording to add NaNs where frames were
                       dropped. This is useful for aligning with sensor data and tracking.

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        return self.readTraces("caTraces", fillDropped)
    
    def readDeconvolvedTraces(self, zScore=False, fillDropped=True):
        '''Read all deconvolved traces from this session. Deconvolved traces are
        a guess of the "true" activity of the neuron.

        Arguments:
        zScore -- If True, z-score the traces per neuron
        fillDropped -- If True, use time index of the recording to add NaNs where frames were
                       dropped. This is useful for aligning with sensor data and tracking.

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        traces = self.readTraces("deconvolvedCaTraces", fillDropped)
        if zScore:
            traces -= traces.mean(axis=0)
            traces /= traces.std(axis=0)
        return traces
    
    def readSensorValues(self, slim=True, onlyRecording=True, reindexFrameNo=True):
        '''Read the sensor values (LEDs and Beams) from this block.
        
        Arguments:
        slim -- If False, also include miscellanous parameters from the session
        onlyRecording -- If True, only return sensor values from when the Ca recording was on
        reindexFrameNo -- Reindex the frameNo variable so that 0 corresponds to the first row
                          of the calcium traces (which is the convention used by findPeaks)
                       
        Returns:
        A Pandas dataframe with the sensor values
        '''
        path = "/sensorValues/{}/{}".format(self.meta.date_rec, self.meta.time_rec)
        sensorValues = pd.read_hdf(self.hdfFile, path)
        
        if slim:
            selectedColumns = ['time', 'trialNo', 'corrTrialNo', 'switchNo', 'rewardNo', 'rewardP',
                               'beamL', 'beamC', 'beamR', 'ledL', 'ledC', 'ledR', 'frameNo']
            sensorValues = sensorValues[selectedColumns]
        
        if onlyRecording:
            sensorValues = sensorValues[sensorValues.frameNo > 0].drop_duplicates("frameNo")
            #First and last frame are sometimes wrong (longer than 50 ms). In that case, throw them.
            if sensorValues.iloc[0].time < sensorValues.iloc[1].time-55:
                sensorValues = sensorValues[1:]
            if sensorValues.iloc[-2].time < sensorValues.iloc[-1].time-55:
                sensorValues = sensorValues[:-1]
        
        if reindexFrameNo:
            sensorValues.frameNo -= sensorValues.frameNo.iloc[0]
        return sensorValues
    
    def findTrials(self, timeColumn="frameNo", onlyRecording=True):
        '''Use the sensor values to find *trials*, i.e. center port entries optionally followed by 
        side port entries. Also indicates whether the trial was initialized correctly and whether it
        was rewarded.
        
        Arguments:
        timeColumn -- The column of sensorValues to use as time measurement. Recommended
                      options are "time" and "frameNo", depending on whether it will
                      later be matched with calcium recordings. Possibly other columns
                      could be used as well (trialNo?), but this is untested.
        onlyRecording -- Only include the part of this block from which there is also calcium video
                         recorded.
                         
        Returns:
        A Pandas Dataframe where each row is a trial attempt.
        
        See also:
        findTrials.findTrials
        '''
        return findTrials.findTrials(self.readSensorValues(onlyRecording=onlyRecording), timeColumn=timeColumn)
    
    def calcActionsPerFrame(self, trials=None):
        '''Reshape the 'trials' dataframe of this block to one row per frame. Useful
        for stretching the behavior before averaging.
        
        Arguments:
        trials -- The trials data frame to use. If None, use the trials from this block.
        
        Returns:
        A dataframe where each row describes the action taken at a given frame.
        '''
        frameNo = 0
        res = []
        if trials is None:
            trials = self.findTrials()
        trials["previousRewarded"] = trials.reward.shift(-1)
        if trials.exitPrevious.iloc[0] > 50000: trials.exitPrevious.iloc[0] = -1
        for t in trials.itertuples():
            while frameNo<t.exitPrevious:
                res.append((frameNo, "other", "none", -1, -1, t.previousRewarded))
                frameNo += 1
            while frameNo<t.enterCenter:
                res.append((frameNo, "sideToCenter", t.previousPort, t.exitPrevious, t.enterCenter, t.previousRewarded))
                frameNo += 1
            while frameNo<t.exitCenter:
                res.append((frameNo, "inPort", "C", t.enterCenter, t.exitCenter, t.reward))
                frameNo += 1
            if t.successfulSide != 1: continue
            while frameNo<t.enterSide:
                res.append((frameNo, "centerToSide", t.chosenPort, t.exitCenter, t.enterSide, t.reward))
                frameNo += 1
            while frameNo<t.exitSide:
                res.append((frameNo, "inPort", t.chosenPort, t.enterSide, t.exitSide, t.reward))
                frameNo += 1
        totLen = self.readSensorValues().shape[0]
        while frameNo<totLen:
            res.append((frameNo, "other", "none", -1, -1, -1))
            frameNo += 1
        res = pd.DataFrame(res, columns=["frameNo", "action", "port", "actionStart", "actionStop", "rewarded"])
        res["actionDuration"] = res.eval("(actionStop - actionStart)")
        res["actionProgress"] = res.eval("(frameNo - actionStart) / actionDuration")
        return res.set_index("frameNo")

def findSessions(hdfFile, onlyRecordedTrials=True, filterQuery=None, **filters):
    store = pd.HDFStore(hdfFile)
    queries = []
    for col, val in filters.items():
        if isinstance(val, str):
            queries.append("{} == '{}'".format(col, val))
        elif isinstance(val, list) or isinstance(val, tuple):
            queries.append("{} in {}".format(col, val))
        else:
            raise ValueError("Unknown filter type")
    meta = pd.read_hdf(store, "/meta")
    if filterQuery is not None: meta = meta.query(filterQuery)
    if queries: meta = meta.query(" & ".join(queries))
    for sessionMeta in meta.itertuples():
        if onlyRecordedTrials and not sessionMeta.caRecordings: continue
        yield Session(store, sessionMeta)