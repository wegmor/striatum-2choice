import pandas as pd
import numpy as np
from deprecated import deprecated
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

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
        path = "/rois/{}/{}/{}".format(self.meta.genotype, self.meta.animal, self.meta.date)
        return pd.read_hdf(self.hdfFile, path)
    
    def readTraces(self, kind="caTraces", fillDropped=True, indicateBlocks=False):
        recordings = []
        for i, rec in enumerate(sorted(self.meta.caRecordings)):
            recData = pd.read_hdf(self.hdfFile, "/caRecordings/{}/{}".format(rec, kind))
            if fillDropped:
                completeIndex = np.arange(recData.index.min(), recData.index.max()+0.025, 0.05)
                recData = recData.reindex(completeIndex, method="nearest", tolerance=1e-3)
            if indicateBlocks:
                recData["block"] = i
                recData = recData.set_index("block", append=True).reorder_levels([1,0])
            recordings.append(recData)
        recData = pd.concat(recordings)
        if str(self) in cutTracesShort:
            recData = recData.iloc[:-cutTracesShort[str(self)]]
        return recData
    
    def readCaTraces(self, fillDropped=True, indicateBlocks=False):
        '''Read all calcium traces from this session.

        Arguments:
        fillDropped -- If True, use time index of the recording to add NaNs where frames were
                       dropped. This is useful for aligning with sensor data and tracking.

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        return self.readTraces("caTraces", fillDropped, indicateBlocks)
    
    def readDeconvolvedTraces(self, zScore=False, rScore=False, fillDropped=True, indicateBlocks=False):
        '''Read all deconvolved traces from this session. Deconvolved traces are
        a guess of the "true" activity of the neuron.

        Arguments:
        zScore -- If True, z-score the traces per neuron
        fillDropped -- If True, use time index of the recording to add NaNs where frames were
                       dropped. This is useful for aligning with sensor data and tracking.

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        traces = self.readTraces("deconvolvedCaTraces", fillDropped, indicateBlocks)
        if zScore:
            traces -= traces.mean(axis=0)
            traces /= traces.std(axis=0)
        if rScore:
            # 10 min zscore window
            window = 10*60*20
            min_window = 4*60*20
            traces -= traces.rolling(window, center=True, min_periods=min_window).mean()
            # v shit neurons may have no signal for 10 minutes -> lots of nan
            traces /= (traces.rolling(window, center=True, min_periods=min_window).std() + 10**-10)
            traces = traces.replace({-np.inf: np.nan, np.inf: np.nan})
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
            if str(self) in cutSensorsShort:
                sensorValues = sensorValues.iloc[:-cutSensorsShort[str(self)]]
            
            #First and last frame are sometimes wrong (longer than 50 ms). In that case, throw them.
            if sensorValues.iloc[0].time < sensorValues.iloc[1].time-55:
                sensorValues = sensorValues[1:]
            if sensorValues.iloc[-2].time < sensorValues.iloc[-1].time-55:
                sensorValues = sensorValues[:-1]
            
        
        if reindexFrameNo:
            sensorValues.frameNo -= sensorValues.frameNo.iloc[0]
        return sensorValues
    
    @deprecated(reason="Please use labelFrameActions instead.")
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
    
    @deprecated(reason="Please use labelFrameActions instead.")
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
        trials["previousRewarded"] = trials.reward.shift(1)
        if trials.exitPrevious.iloc[0] > 30000: trials.exitPrevious.iloc[0] = -1
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
    
    def labelFrameActions(self, sensorValues=None, reward="returns", switch=False, splitCenter=True):
        '''Assign a string code to every frame, indicating where in the task the mouse currently is.
    
        Arguments:
        sensorValues --- A Pandas Dataframe as given by block.readSensorValues()
        rewards --- Where to indicate rewards / omissions in the codes. Can be "never" to never show reward info,
                    "sidePorts" to only show it in the side ports, "returns" to show it for ports AND return movements
                    or "fullTrial" to always show it.
        switches -- Whether to indicate stay / switch trials. In the output, stays are indicated by "." 
                    and switches by "!".
        '''
        if sensorValues is None:
            sensorValues = self.readSensorValues()
        return findTrials.labelFrameActions(sensorValues, reward, switch, splitCenter)
    
    def getRewSwDist(self):
        apf = self.labelFrameActions(reward='sidePorts', switch=True).reset_index(drop=True)
        apf.index.name = 'frameNo'
        
        actions = apf.reset_index('frameNo').groupby('actionNo').first()
        actions['switch'] = actions.label.str.contains('p[RL]2.d!').astype('int')
        actions['reward'] = actions.label.str.contains('p[RL]2.r[.!]').astype('int')
        actions['switchNo'] = actions.switch.cumsum()
        actions['rewardNo'] = actions.reward.cumsum().shift(-1)
        actions = actions.loc[actions.label.str.contains('p[RL]2.d[.!]')]
    
        # count trials without a reward
        actions['sinceReward'] = actions.groupby('rewardNo').cumcount()
        actions['toReward'] = -(actions.groupby('rewardNo').cumcount(ascending=False).shift(1))
        actions.loc[actions.rewardNo == 0, 'sinceReward'] = np.nan
        actions.loc[actions.rewardNo == actions.rewardNo.iloc[-1], 'toReward'] = np.nan
        actions.loc[actions.sinceReward == 0, 'toReward'] = 0  # otherwise last switch is lost in "toReward"
        
        # count trials without switch
        actions['sinceSwitch'] = actions.groupby('switchNo').cumcount()
        actions['toSwitch'] = -(actions.groupby('switchNo').cumcount(ascending=False).shift(1))
        actions.loc[actions.switchNo == 0, 'sinceSwitch'] = np.nan
        actions.loc[actions.switchNo == actions.switchNo.iloc[-1], 'toSwitch'] = np.nan
        actions.loc[actions.sinceSwitch == 0, 'toSwitch'] = 0
        
        # insert into apf
        actions = actions.fillna(-999999) # otherwise fillna below causes trouble
        actions.set_index('frameNo', inplace=True)
        fields = ['sinceReward','toReward','sinceSwitch','toSwitch','switchNo','rewardNo']
        apf[fields] = actions[fields]
        apf[fields] = apf[fields].fillna(method='ffill')
        apf[fields[:-2]] = apf[fields[:-2]].replace(-999999, np.nan)
        
        return apf[fields].reset_index(drop=True).copy()

    def readTracking(self, inCm=False):
        tracking = pd.read_hdf(self.hdfFile, "/tracking/" + self.meta.video)
        if inCm:
            if self.meta.cohort == "2018":
                raise ValueError("No corners have been added for the 2018 cohort")
            if self.meta.task == "openFieldAgain":
                raise ValueError("No corners have been added for the second open field sessions")
            boxCorners = pd.read_hdf(self.hdfFile, "/extra/boxCorners")
            boxSize = 49 if self.meta.task == "openField" else 15
            tracking = perspectiveTransform(tracking, boxCorners.loc[self.meta.video], boxSize, boxSize)
        if self.meta.cohort=="2018" and hasEmptyFirstFrame[str(self)]:
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
        if str(self) in cutTrackingShort and self.meta.task=="2choice":
            tracking = tracking.iloc[:-cutTrackingShort[str(self)]]
        return tracking
    
    def shuffleFrameLabels(self, n=1, switch=True, reward='sidePorts',
                           splitCenter=True):
        frameLabels = self.labelFrameActions(reward=reward, splitCenter=splitCenter,
                                             switch=True)
        frameLabels.index.name = 'frame'
        frameLabels["actionFrame"] = (frameLabels.actionDuration * frameLabels.actionProgress).astype(np.int64)
        
        actions = frameLabels.reset_index().groupby('actionNo').first()
        switch_idx = ((actions.label.shift(1).str.contains('d[RL]2.[or]?[\.!]?') == True) 
                       & actions.label.str.endswith('!'))
        switchFrames = actions.loc[switch_idx, 'frame'].values
        frameLabels.loc[switchFrames, 'switch'] = 1
        frameLabels['switch'] = frameLabels.switch.fillna(0).cumsum()
        
        # test if every second block is a left/right block (thus, blocks alternate)
        if not len(np.unique(frameLabels.groupby('switch').first().label.str.contains('^.L')
                                        .astype('int').values[1::2])) == 1:
            print('/!\ left and right blocks do not appear to alternate!')
        
        sidx1_orig = frameLabels.switch.unique()[::2]
        sidx1_shuffle = sidx1_orig.copy()
        sidx2_orig = frameLabels.switch.unique()[1::2]
        sidx2_shuffle = sidx2_orig.copy()
        
        labels_shuffled = []
        for _ in range(n):
            fl_shuffled = frameLabels.copy()
            np.random.shuffle(sidx1_shuffle)
            np.random.shuffle(sidx2_shuffle)
            
            replace_dict = dict(list(zip(sidx1_orig, sidx1_shuffle)) +
                                list(zip(sidx2_orig, sidx2_shuffle)))
            fl_shuffled['switch'] = fl_shuffled.switch.replace(replace_dict)
            fl_shuffled = fl_shuffled.sort_values(['switch','actionNo','actionFrame'])
            
            replace_dict = dict(zip(fl_shuffled.actionNo.unique(),
                                    np.arange(len(fl_shuffled.actionNo.unique()))))
            fl_shuffled['actionNo'] = fl_shuffled.actionNo.replace(replace_dict)
            
            fl_shuffled = fl_shuffled.reset_index(drop=True).copy()
            
            if not switch:
                fl_shuffled['label']  = fl_shuffled.label.str.slice(0,-1)
                
            labels_shuffled.append(fl_shuffled[['label','actionNo','actionFrame',
                                                'actionDuration','actionProgress']])
        
        return(labels_shuffled)

    def getWallCorners(self):
        '''Get the coordinates of the wall of the box. No calculatation is performed, values
        are hard-coded.

        Returns:
        2018 cohort: A list of length 4 with the coordinates (left, bottom, right, top)
        2019 cohort: A pandas Series with a pair of coordinates for each corner (box is not
                     necessarily square)
        '''
        if self.meta.cohort == "2019":
            return pd.read_hdf(self.hdfFile, "/extra/boxCorners").loc[self.meta.video]
        else:
            return wallCorners[str(self)]
    
    def _repr_html_(self):
        output = "<table><thead><tr><th>{}</th><th></th></tr></thead><tbody>".format(str(self))
        order = ("genotype", "animal", "date", "task", "cohort",
                 "camera", "board", "video", "caRecordings")
        for key in order:
            output += "<tr><th>{}</th><td>{}</td>".format(key, self.meta.__getattribute__(key))
        output += "</tbody></table>"
        return output

def findSessions(hdfFile, onlyRecordedTrials=True, filterQuery=None, sortBy=None, closeStore=True, **filters):
    store = pd.HDFStore(hdfFile, 'r')
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
    if sortBy is not None: meta = meta.sort_values(sortBy)
    for sessionMeta in meta.itertuples():
        if onlyRecordedTrials and not sessionMeta.caRecordings: continue
        yield Session(store, sessionMeta)
    if closeStore: store.close()

def perspectiveTransform(tracking, corners, boxW=49, boxH=49):
    import cv2
    src = corners.unstack().loc[["lowerLeft", "upperLeft", "upperRight", "lowerRight"]]
    src = np.array(src, dtype=np.float32)
    dst = np.array([(0,0), (0, boxH), (boxW, boxH), (boxW, 0)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src, dst)
    res = dict()
    for bodyPart in tracking.columns.levels[0]:
        if bodyPart == "block": continue
        homTransformed = transform.dot(tracking[bodyPart][["x", "y"]].assign(t=1).T)
        homTransformed /= homTransformed[2,:]
        res[(bodyPart, "x")] = homTransformed[0,:]
        res[(bodyPart, "y")] = homTransformed[1,:]
        res[(bodyPart, "likelihood")] = tracking[bodyPart].likelihood
    res = pd.DataFrame(res, index=tracking.index)
    if "block" in tracking.columns:
        res.insert(0, "block", tracking.block)
    return res
    
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

cutSensorsShort = {
    'd1_5652_190203': 5,
    'd1_5643_190114': 5,
    'a2a_6043_190126': 84
}

cutTracesShort = {
    'a2a_6043_190126': 84,
    'oprm1_5308_190205': 88
}

cutTrackingShort = {
    'a2a_6043_190126': 84
}


#TODO: move these to HDF file
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
