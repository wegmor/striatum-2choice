import pandas as pd
import numpy as np

import pyximport
pyximport.install()
from . import findPeaks
from . import findTrials


class Block:
    '''A block is one recording session.'''
    def __init__(self, hdfFile, path):
        self.hdfFile = hdfFile
        _, _, self.genotype, self.mouseNumber, self.date, self.recording, _ = path.split('/')
    
    def __str__(self):
        return self.genotype + "_" + self.mouseNumber + "_"+self.date

    def readROIs(self):
        '''Read the regions-of-interest (ROIs) for each neuron in this block. Contains the shape
        of the neurons.

        Returns:
        A Pandas panel with all the ROIs stacked on the item axis        
        '''
        return pd.read_hdf(self.hdfFile, self.getPath()+"neuronsROI")
    
    def readCaTraces(self, dropFirst=0):
        '''Read all calcium traces from this block.

        Arguments:
        dropFirst -- Number of seconds to drop from the beginning

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        path = self.getPath()+self.recording+"/"+'caTraces'
        caTraces = pd.read_hdf(self.hdfFile, path)
        return caTraces[dropFirst:]
    
    def readDeconvolvedTraces(self, dropFirst=0):
        '''Read all deconvolved traces from this block. Deconvolved traces are
        a guess of the "true" activity of the neuron.

        Arguments:
        dropFirst -- Number of seconds to drop from the beginning

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        path = self.getPath()+self.recording+"/"+'deconvolvedCaTraces'
        traces = pd.read_hdf(self.hdfFile, path)
        return traces[dropFirst:]
    
    def getPath(self):
        '''The path in the HDF file corresponding to this block.'''
        return "/data/"+self.genotype+"/"+self.mouseNumber+"/"+self.date+"/"
    
    def findPeaks(self, method="donahue", shape="wide"):
        '''Find "peaks" in the calcium traces.
        
        Arguments:
        method --- The method used to find the peaks
        shape --- Either "wide" or "long". Wide form is compatible with
                  caTraces and deconvolvedTraces, while long form is more
                  memory efficient for this sparse signal.
        '''
        return findPeaks.findPeaks(self.readCaTraces(), method=method, shape=shape)
    
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
        
        sensorValues = pd.read_hdf(self.hdfFile, self.getPath() + 'sensors')
        
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
    
    def readTracking(self):
        tracking = pd.read_hdf(self.hdfFile, self.getPath()+'tracking')
        if hasEmptyFirstFrame[str(self)]:
            tracking = tracking.iloc[1:]
            tracking.index.name = "videoFrameNo"
            tracking.reset_index(inplace=True)
        
        #Special cases to fix wrong number of frames
        if str(self) == "d1_3517_180329":
             #First frame is dark and from LED intensities it looks like it should be dropped
            tracking = tracking.iloc[1:-1]
            tracking.index.name = "videoFrameNo"
            tracking.reset_index(inplace=True)
        elif str(self) == "oprm1_3582_180327":
            #From LED it looks like first two frames are missing
            #tracking.insert(0, {c: np.nan for c in tracking.columns})
            tracking = tracking.reindex(np.arange(-2, len(tracking)))
            tracking.index.name = "videoFrameNo"
            tracking.reset_index(inplace=True)
            
        return tracking

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
        
    def calcTunings(self, event="exitSide", splitCriterion="reward==True", windowSize=10, sampleBy=["chosenPort", "reward"]):
        '''Calculate tunings of each neuron in this block using the Donahue method. 
        
        Arguments:
        event -- The event to which to align the activity. Should be one of "enterCenter",
                 "exitCenter", "enterSide", "exitSide".
        splitCriterion -- The criterion to compare, will be passed on to pandas.DataFrame.eval
                          and should be a boolean expression.
        windowSize -- The duration after the event during which to look for peaks, specified as an
                      integer number of frames. Each frame is 50ms, thus default 10 frames is 500ms.
        sampleBy -- To compensate for different number of trials, subsample the set of trials so
                    that all combinations of these columns occur equally often. If None, don't subsample.
                    
        Returns:
        A numpy array with the Z-statistic for each individual neuron
        '''
        import statsmodels.stats.proportion
        
        peaks = self.findPeaks()
        trials = self.findTrials()
        trials.query("successfulEnd==1", True)
        trials.reward = trials.reward > 0
        
        if sampleBy is not None:
            cases = trials.groupby(sampleBy)
            #The smallest case
            minNum = cases.size().min()
            #Sample minNum for all cases
            trials = cases.apply(pd.DataFrame.sample, minNum).reset_index(level=[0,1], drop=True).sort_index()
            
        avgPerTrial = pd.concat([peaks.iloc[i:i+windowSize].sum() > 0 for i in trials[event]], 1).T
        avgPerTrial.index = trials.index.copy()

        split = trials.eval(splitCriterion)
        A = avgPerTrial[split==True].sum()
        B = avgPerTrial[split==False].sum()

        test = statsmodels.stats.proportion.proportions_ztest
        tunings = np.array([test(p, (len(A), len(B)))[0] for p in zip(A,B)])

        return tunings
    
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
    
    def getWallCorners(self):
        '''Get the coordinates of the wall of the box. No calculatation is performed, values
        are hard-coded.

        Returns:
        A list of length 4 with the coordinates (left, bottom, right, top)
        '''
        return wallCorners[str(self)]

def findBlocks(hdfFile, onlyRecordedTrials=True, genotype=None, mouseNumber=None, date=None, recording=None):
    '''Generator of all experimental blocks stored in a HDF file.
    
    Arguments:
    hdfFile -- The filename of the HDF file
    genotype, mouseNumber, date -- Filters for selecting sessions. Specify as None (no filter, default),
                                   a specific value (string) or a list of values (any iterable)
    Example:
    >> for b in findBlocks('endoData.hdf', genotype=['a2a', 'oprm1']):
    >>     print b.genotype, b.mouseNumber, b.readROIs().shape
    '''
    store = pd.HDFStore(hdfFile)
    for k in store.keys():
        if k.endswith('/caTraces' if onlyRecordedTrials else '/sensors'):
            s = Block(store, k if onlyRecordedTrials else k[:-7]+'None/None')
            genOk = genotype is None or s.genotype == genotype or s.genotype in genotype
            numOk = mouseNumber is None or s.mouseNumber == str(mouseNumber) or s.mouseNumber in map(str, mouseNumber)
            datOk = date is None or s.date == date or s.date in date
            recOk = recording is None or s.recording == recording or s.recording in recording
            if all((genOk, numOk, datOk, recOk)):
                yield s


#I manually checked whether the first frame is an empty one for each video:          
hasEmptyFirstFrame = {
    'd1_3517_180404':    False,
    'd1_3517_180329':    False,
    'a2a_3241_180405':   True,
    'a2a_3241_180403':   False,
    'a2a_3241_180326':   True,
    'a2a_3242_180330':   False,
    'a2a_3244_180410':   False,
    'a2a_3244_180405':   True,
    'a2a_3244_180330':   True,
    'a2a_3245_180405':   True,
    'a2a_3245_180410':   False,
    'a2a_3245_180403':   False,
    'oprm1_3323_180409': False,
    'oprm1_3321_180409': False, #Actually True, but LED says it's aligned anyways
    'oprm1_3517_180403': True,
    'oprm1_3582_180404': False,
    'oprm1_3581_180402': False,
    'oprm1_3323_180331': False,
    'oprm1_3321_180331': False,
    'oprm1_3582_180329': False,
    'oprm1_3572_180329': False,
    'oprm1_3572_180403': True,
    'oprm1_3582_180327': False,
    'oprm1_3323_180327': False,
    'oprm1_3321_180327': False
}


wallCorners = {
 'a2a_3241_180326': [88.333, 252.667, 289.66700000000003, 49.667],
 'a2a_3241_180403': [94.333, 253.333, 296.0, 48.333],
 'a2a_3244_180330': [93.667, 252.667, 295.66700000000003, 48.667],
 'a2a_3244_180405': [93.0, 252.667, 294.33299999999997, 48.667],
 'a2a_3245_180405': [93.0, 253.333, 293.33299999999997, 48.667],
 'a2a_3245_180410': [80.667, 253.667, 280.66700000000003, 49.667],
 'd1_3517_180329': [94.333, 252.667, 295.33299999999997, 49.0],
 'd1_3517_180404': [94.333, 250.667, 295.0, 47.667],
 'oprm1_3321_180327': [100.333, 252.0, 302.33299999999997, 48.667],
 'oprm1_3321_180409': [92.667, 252.0, 293.66700000000003, 48.667],
 'oprm1_3323_180327': [103.0, 253.0, 304.33299999999997, 48.667],
 'oprm1_3323_180409': [104.667, 252.0, 307.33299999999997, 47.333],
 'oprm1_3572_180329': [94.0, 252.667, 295.66700000000003, 49.667],
 'oprm1_3572_180403': [93.333, 253.667, 296.0, 48.0],
 'oprm1_3582_180327': [103.333, 252.667, 304.33299999999997, 48.333],
 'oprm1_3582_180329': [93.333, 252.333, 296.0, 49.0]
}