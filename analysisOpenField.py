import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sklearn.svm
import sklearn.ensemble
import sklearn.model_selection
import tqdm
import multiprocessing
import functools
import pyximport; pyximport.install()
import scipy.stats

from utils import readSessions, particleFilter, segmentBehaviors
from utils.cachedDataFrame import cachedDataFrame

@cachedDataFrame("filteredOpenField.pkl")
def filterAllOpenField(dataFile):
    all_filtered = []
    for sess in readSessions.findSessions(dataFile, task="openField"):
        deconv = sess.readDeconvolvedTraces(indicateBlocks=True)
        tracking = sess.readTracking(inCm=True)
        tracking.index = deconv.index
        blocks = tracking.index.levels[0]
        filtered = []
        for block in blocks:
            t = tracking.loc[block]
            filtered.append(particleFilter.particleFilter(t, nParticles=2000, flattening=1e-12))
        filtered = pd.concat(filtered)
        filtered.rename(columns={"bodyAngle": "bodyDirection"}, inplace=True)
        filtered.rename_axis("time", axis=0, inplace=True)
        filtered.bodyDirection *= 180/np.pi
        ind = tracking.index.to_frame()
        ind.insert(0, "session", str(sess))
        filtered.index = pd.MultiIndex.from_frame(ind)
        all_filtered.append(filtered)
    return pd.concat(all_filtered)

@cachedDataFrame("segmentedBehavior.pkl")
def segmentAllOpenField(dataFile):
    all_filtered = filterAllOpenField(dataFile)
    all_segmented = []
    for sess, outer in all_filtered.groupby(level=0):
        print(str(sess))
        action_no = 0
        frame_no = 0
        for block, filtered in outer.groupby(level=1):
            segmented = segmentBehaviors.segmentBehaviors(filtered)
            segmented.insert(0, "session", sess)
            segmented.insert(1, "block", block)
            segmented.insert(2, "actionNo", action_no + np.arange(len(segmented)))
            action_no += len(segmented)
            segmented.startFrame += frame_no
            segmented.stopFrame += frame_no
            frame_no = segmented.stopFrame.iloc[-1]
            all_segmented.append(segmented)
    return pd.concat(all_segmented).set_index(["session", "block", "actionNo"])

def _crossValScore(X, Y):
    svm = sklearn.svm.SVC(kernel="linear", cache_size=2000)
    trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, stratify=Y)
    svm.fit(trainX, trainY)
    predicted = svm.predict(testX)
    accuracy = np.mean(predicted == testY)
    return accuracy

def _testRealAndShuffled(i, realX, realY, shuffledX, shuffledY, nNeurons):
    np.random.seed(np.random.randint(1000000)+i) #Seed each process differently
    neurons = np.random.choice(realX.shape[1], nNeurons, replace=False)
    realScore = _crossValScore(realX[neurons], realY)
    shuffledScore = _crossValScore(shuffledX[neurons], shuffledY)
    return (i, realScore, shuffledScore)

def _prepareTrials(deconv, behaviors, minDuration=20):
    avgSig = deconv.groupby(behaviors.actionNo).mean()
    labels = behaviors.groupby("actionNo").first().behavior
    duration = behaviors.groupby("actionNo").size()
    validTrials = np.logical_and(avgSig.notna().all(axis=1), duration >= minDuration)
    X = avgSig[validTrials]
    Y = labels[validTrials]
    return X, Y

@cachedDataFrame("openFieldDecodingWithIncreasingNumberOfNeurons.pkl")
def decodeWithIncreasingNumberOfNeurons(dataFile):
    allBehaviors = segmentAllOpenField(dataFile)
    nShufflesPerNeuronNum = 20
    with multiprocessing.Pool(5) as pool:
        res = []
        for sess in readSessions.findSessions(dataFile, task="openField"):
            deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
            behaviors = allBehaviors.loc[str(sess)].reset_index()
            behaviors = behaviors.set_index("startFrame", drop=True)[["actionNo", "behavior"]]
            behaviors = behaviors.reindex(np.arange(len(deconv)), method="ffill")
            if behaviors.behavior.nunique() < 4: continue
            suffledBehaviors = behaviors.copy()
            suffledBehaviors["behavior"] = np.random.permutation(suffledBehaviors.behavior)
            realX, realY = _prepareTrials(deconv, behaviors)
            shuffledX, shuffledY = _prepareTrials(deconv, suffledBehaviors)
            with tqdm.tqdm(total=int(realX.shape[1]/5)*nShufflesPerNeuronNum, desc=str(sess)) as t:
                for nNeurons in range(5, realX.shape[1], 5):
                    fcn = functools.partial(_testRealAndShuffled, realX=realX, realY=realY,
                                            shuffledX=shuffledX, shuffledY=shuffledY, nNeurons=nNeurons)
                    for scores in pool.imap(fcn, range(nShufflesPerNeuronNum)):
                        res.append((str(sess), sess.meta.task, nNeurons)+scores)
                        t.update(1)
    return pd.DataFrame(res, columns=["session", "task", "nNeurons", "i", "realAccuracy", "shuffledAccuracy"])

@cachedDataFrame("openFieldDecodingConfusion.pkl")
def decodingConfusion(dataFile):
    allBehaviors = segmentAllOpenField(dataFile)
    confMats = []
    for sess in readSessions.findSessions(dataFile, task="openField"):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        behaviors = allBehaviors.loc[str(sess)].reset_index()
        behaviors = behaviors.set_index("startFrame", drop=True)[["actionNo", "behavior"]]
        behaviors = behaviors.reindex(np.arange(len(deconv)), method="ffill")
        realX, realY = _prepareTrials(deconv, behaviors)
        for i in tqdm.trange(5, desc=str(sess)):
            trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(realX, realY,
                                                                                test_size=0.2, stratify=realY)
            svm = sklearn.svm.SVC(kernel="linear").fit(trainX, trainY)
            pred = svm.predict(testX)
            m = sklearn.metrics.confusion_matrix(testY, pred)
            m = pd.DataFrame(m, index=svm.classes_, columns=svm.classes_)
            m = m.rename_axis(index="true", columns="predicted").unstack()
            m = m.rename("occurencies").reset_index()
            m["sess"] = str(sess)
            m["i"] = i
            m["nNeurons"] = deconv.shape[1]
            confMats.append(m)
    return pd.concat(confMats)




def getActionAverages(traces, behaviors):
    actionAvg = traces.groupby([behaviors.behavior, behaviors.actionNo]).mean().dropna()
    labels = behaviors.groupby(['behavior','actionNo']).behavior.first()
    return(actionAvg, labels)

def wAvg(group, var, weights):
    return(np.average(group[var], weights=group[weights]))
    
def bootstrap(group, var, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(group[var]), size=len(group[var]),
                               replace=True)
        avgs.append(np.average(group[var].iloc[idx], weights=group[weights].iloc[idx]))
    return(np.std(avgs))
    
def jitter(x, std):
    return(x+np.random.normal(0,std,size=len(x)))
    
def shuffleBehaviors(behaviors):
    shuffled = behaviors.copy()
    shuffled["behavior"] = np.random.permutation(shuffled.behavior)
    return shuffled

#%%
@cachedDataFrame("openFieldTunings.pkl")
def getTuningData(dataFilePath, no_shuffles=1000):
    allBehaviors = segmentAllOpenField(dataFilePath)
    df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='openField'):
        traces = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True) # frame no as index
        behaviors = allBehaviors.loc[str(s)].reset_index()
        behaviors = behaviors.set_index("startFrame", drop=False)[["actionNo", "behavior"]]
        behaviors = behaviors.reindex(np.arange(len(traces)), method="ffill")
        if behaviors.behavior.nunique() < 4:
            raise ValueError("All four actions are not present in the session {}.".format(s))
        actionAvg, labels = getActionAverages(traces, behaviors)  # mean per action
        actionAvg = actionAvg.groupby('behavior').mean()       # mean per label
        
        s_actionAvgs = []
        for i in tqdm.trange(no_shuffles, desc=str(s)):
            s_behaviors = shuffleBehaviors(behaviors)
            s_actionAvg, s_labels = getActionAverages(traces, s_behaviors)
            s_actionAvg = s_actionAvg.groupby('behavior').mean()
            s_actionAvgs.append(s_actionAvg)
        s_actionAvgs = pd.concat(s_actionAvgs, keys=np.arange(no_shuffles),
                                 names=['shuffle']).reorder_levels(['behavior','shuffle'])

        for action, adata in s_actionAvgs.groupby('behavior'):
            for neuron in adata:
                ndict = {}
                dist = adata[neuron].values # shuffled "label" means distribution
                value = actionAvg.loc[action, neuron] # actual mean
                
                ndict['genotype'] = s.meta.genotype
                ndict['animal'] = s.meta.animal
                ndict['date'] = s.meta.date
                ndict['neuron'] = neuron
                ndict['action'] = action
                ndict['mean'] = value
                ndict['s_mean'] = dist.mean()
                ndict['s_std'] = dist.std()
                ndict['tuning'] = (ndict['mean'] - ndict['s_mean']) / ndict['s_std']
                # v percentile of the actual mean in the shuffled distribution
                ndict['pct'] = np.searchsorted(np.sort(dist), value) / len(dist)
                
                df = df.append(pd.Series(ndict), ignore_index=True)
        
    return df


class BlockKFold(sklearn.model_selection.BaseCrossValidator):
    def __init__(self, nFolds=5, samplesPerGroup=100):
        self.nFolds = 5
        self.samplesPerGroup = samplesPerGroup
        
    def _iter_test_masks(self, X=None, y=None, groups=None):
        nSamples = X.shape[0]
        nGroups = np.int(np.ceil(nSamples / self.samplesPerGroup))
        perm = np.random.permutation(nGroups)
        for i in range(self.nFolds):
            test_mask = np.zeros(nSamples, dtype=np.bool)
            for j in range(i, nGroups, self.nFolds):
                lo = perm[j]*self.samplesPerGroup
                hi = min(lo+self.samplesPerGroup, nSamples)
                test_mask[lo:hi] = True
            yield test_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.nFolds

def decodeWallAngle(dataFilePath):
    all_dfs = []
    for sess in readSessions.findSessions(dataFilePath, task="openField"):
        print(str(sess))
        
        #TODO: Replace with particle filter coordinates
        tracking = sess.readTracking(inCm=True)
        likelihood = tracking[[("leftEar", "likelihood"),
                               ("rightEar", "likelihood"),
                               ("tailBase", "likelihood")]].min(axis=1)
        coords = 0.5*(tracking.leftEar + tracking.rightEar)
        wallDists = pd.concat((coords.x, coords.y, 49-coords.x, 49-coords.y), axis=1)
        wallDists.columns = ["left", "bottom", "right", "top"]
        closestWallId = wallDists.idxmin(axis=1)
        bodyVec = coords - tracking.tailBase
        bodyDir = np.arctan2(bodyVec.y, bodyVec.x).rename("bodyDirection")
        angleOfWall = closestWallId.replace({'left': np.pi/2, 'top': 0,
                                             'right': -np.pi/2, 'bottom': np.pi})
        wallAngle = (angleOfWall - bodyDir + 2*np.pi)%(2*np.pi) - np.pi
        
        #Distance to the closest wall
        minWallDist = wallDists.min(axis=1)
        
        #Coordinates of the wall in egocentric coordinates
        wall_xx = np.cos(wallAngle)*minWallDist
        wall_yy = np.sin(wallAngle)*minWallDist

        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        
        #Select good frames to use
        mask = minWallDist < 5 #Head closer than 5 cm to the wall
        mask[deconv.isna().any(axis=1)] = False #Discard frames where neural data is missing
        mask[likelihood<0.9] = False #Discard frames where DeepLabCut is uncertain
        
        #Setup in the format expected by scikit learn
        X = deconv[mask]
        Y_xx = wall_xx[mask]
        Y_yy = wall_yy[mask]
        
        svr = sklearn.svm.SVR("linear")
        cv = BlockKFold(5, 100)#sklearn.model_selection.KFold(5)
        pred_xx = sklearn.model_selection.cross_val_predict(svr, X, Y_xx, cv=cv, n_jobs=5)
        pred_yy = sklearn.model_selection.cross_val_predict(svr, X, Y_yy, cv=cv, n_jobs=5)
        shufflePerm = np.random.permutation(X.shape[0])
        shuffled_xx = sklearn.model_selection.cross_val_predict(svr, X.iloc[shufflePerm, :], Y_xx, cv=cv, n_jobs=5)
        shuffled_yy = sklearn.model_selection.cross_val_predict(svr, X.iloc[shufflePerm, :], Y_yy, cv=cv, n_jobs=5)
        df = pd.DataFrame({'sess': str(sess), 'x_true': Y_xx, 'y_true': Y_yy,
                           'x_predicted': pred_xx, 'y_predicted': pred_yy,
                           'x_shuffled_prediction': shuffled_xx,
                           'y_shuffled_prediction': shuffled_yy,
                           'nNeurons': X.shape[1], 'nFrames': X.shape[0]})
        all_dfs.append(df)
    return pd.concat(all_dfs)

@cachedDataFrame("openFieldSpeedTunings.pkl")
def calculateSpeedTuning(dataFilePath, nShuffles=1000):
    behaviors = segmentAllOpenField(dataFilePath)
    allDfs = []
    for sess in readSessions.findSessions(dataFilePath, task="openField"):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        tracking = sess.readTracking(inCm=True)
        if len(deconv) != len(tracking): continue
        coords = particleFilter.particleFilter(tracking, flattening = 1e-12)
        df = behaviors.query("behavior=='running' & session=='{}'".format(sess))
        runningFrames = np.concatenate([np.arange(r.startFrame, r.stopFrame)
                                        for r in df.itertuples()])
        speed = coords.iloc[runningFrames].speed
        realCorr = deconv.iloc[runningFrames].corrwith(speed)
        shuffles = []
        for i in tqdm.trange(nShuffles):
            shuffledSpeed = pd.Series(np.random.permutation(speed), speed.index)
            shuffles.append(deconv.iloc[runningFrames].corrwith(shuffledSpeed))
        shuffles = pd.concat(shuffles, axis=1)
        pct = np.array([np.searchsorted(a, r)
                        for r, a in zip(realCorr, shuffles.values)]) / nShuffles
        tuning = (realCorr - shuffles.mean(axis=1)) / shuffles.std(axis=1)
        allDfs.append(pd.DataFrame({'sess': str(sess),
                                    'neuron': deconv.columns,
                                    'correlation': realCorr,
                                    'pct': pct,
                                    'tuning': tuning}))
    return pd.concat(allDfs)

@cachedDataFrame("oft_illustration.pkl")
def getSmoothedOFTTracking(dataFile, genotype, animal, date):
    sess = next(readSessions.findSessions(dataFile, task="openField",
                                          genotype=genotype, animal=animal,
                                          date=date))
    tracking = sess.readTracking(inCm=True)
    coords = particleFilter.particleFilter(tracking, flattening = 1e-12)
    coords.rename(columns={"bodyAngle": "bodyDirection"}, inplace=True)
    coords.rename_axis("time", axis=0, inplace=True)
    coords.bodyDirection = np.rad2deg(coords.bodyDirection)
    behaviors = segmentBehaviors.segmentBehaviors(coords)[['startFrame','behavior']]
    coords.reset_index(inplace=True)
    coords.rename(columns={'time':'frame'}, inplace=True)
    behaviors.insert(0, "actionNo", behaviors.index.copy())
    coords = coords.merge(behaviors, left_on='frame', right_on='startFrame',
                          how='left').fillna(method='ffill')
    return coords

@cachedDataFrame("avgActivityPerSpeed.pkl")
def avgActivityPerSpeed(dataFilePath):
    all_filtered = filterAllOpenField(dataFilePath)
    bins = [-.00001, .1, 1, 2, 3, 5, 8, 10]
    per_session = []
    meta = []
    for sess in readSessions.findSessions(dataFilePath, task="openField"):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        tracking = sess.readTracking(inCm=True)
        speed = all_filtered.loc[str(sess)].reset_index().speed*20
        speed_bin = pd.cut(speed, bins).cat.codes.astype("int")
        pop_activity = deconv.mean(axis=1)
        activity_per_bin = pop_activity.groupby(speed_bin).mean().sort_index()
        per_session.append(activity_per_bin)
        meta.append((str(sess), deconv.shape[1]))
    meta = pd.MultiIndex.from_tuples(meta, names=["session", "noNeurons"])
    return pd.DataFrame(per_session, index=meta)

@cachedDataFrame("openFieldEventWindows.pkl")
def getEventWindows(endoDataPath, events, win_size=(20, 19)):
    segmented = segmentAllOpenField(endoDataPath)
    windows = pd.DataFrame()
    for s in readSessions.findSessions(endoDataPath, task='openField'):
        deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        thisSegmented = segmented.loc[str(s)].copy()
        thisSegmented.index = thisSegmented.index.droplevel(0)
        lastFrame = thisSegmented.stopFrame.iloc[-1]
        segs = thisSegmented.reset_index().set_index("startFrame").reindex(np.arange(lastFrame), method="ffill")
        segs['index'] = deconv.index
        deconv = deconv.set_index([segs.behavior,segs.actionNo], append=True)
        deconv.index.names = ['index','label','actionNo']
        deconv.columns.name = 'neuron'      
        center_idx = (segs.loc[segs.behavior.isin(events)].groupby('actionNo')
                              [['index','actionNo','behavior']].first().values)
        _windows = []
        neurons = deconv.columns
        for idx, actionNo, label in tqdm.tqdm(center_idx, desc=str(s)):
            win = deconv.loc[idx-win_size[0]:idx+win_size[1]].reset_index()
            win.loc[win.actionNo > actionNo, neurons] = np.nan
            win['frameNo'] = np.arange(len(win))
            win['label'] = label
            win['actionNo'] = actionNo
            win = win.set_index(['actionNo','label','frameNo'])[neurons]
            win = win.unstack('frameNo').stack('neuron')
            win.columns = pd.MultiIndex.from_product([['frameNo'], win.columns])
            _windows.append(win.reset_index())
        _windows = pd.concat(_windows, ignore_index=True)
        
        for k,v in [('date',s.meta.date),('animal',s.meta.animal),('genotype',s.meta.genotype)]:
            _windows.insert(0,k,v)
        windows = windows.append(_windows, ignore_index=True)
    return windows

from scipy.spatial.distance import pdist, squareform

@cachedDataFrame("tuning_pdists_open_field.pkl")
def getPDistData(dataFilePath, tuningData, no_shuffles=1000):
    dist_df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='openField'):
        # load ROI centers
        roics = np.array(list(s.readROIs().idxmax()))
        # generate shuffled ROIs
        roics_shuffle = [np.random.permutation(roics) for _ in range(no_shuffles)]
        
        # calc pairwise distances
        # inscopix says 1440 px -> 900 um; 4x downsampled 900/360 = 2.5
        dist = squareform(pdist(roics)) * 2.5
        dist[np.diag_indices_from(dist)] = np.nan
    
        # load tuning data for session
        tunings = tuningData.query('animal == @s.meta.animal & date == @s.meta.date').copy()
        if len(tunings) == 0: continue
        tunings = tunings.set_index(['action','neuron']).sort_index()
        
        min_dists = []
        min_dists_shuffle = []
        for action, ts in tunings.groupby('action'):
            if ts.signp.sum() >= 2: # at least 2 tuned neurons in this group?
                # find the minimum distance to the closest neuron tuned to action
                min_dists += np.nanmin(dist[ts.signp][:,ts.signp], axis=0).tolist()
                
                # calculate min distance for shuffled ROIs
                for roicss in roics_shuffle:
                    dist_shuffle = squareform(pdist(roicss)) * 2.5
                    dist_shuffle[np.diag_indices_from(dist_shuffle)] = np.nan
                    min_dists_shuffle += np.nanmin(dist_shuffle[ts.signp][:,ts.signp], axis=0).tolist()
        
        # calculate mean minimum distance, real & expected by chance, for session
        mean_dist = np.mean(min_dists)
        mean_dist_shuffle = np.mean(min_dists_shuffle)
    
        series = pd.Series({'genotype': s.meta.genotype,
                            'animal': s.meta.animal, 'date': s.meta.date,
                            'dist': mean_dist, 'dist_shuffle': mean_dist_shuffle,
                            'noNeurons': len(ts)})
        dist_df = dist_df.append(series, ignore_index=True)
        
    return dist_df

@cachedDataFrame("openFieldPopulationSpeedCorr.pkl")
def populationSpeedCorr(endoDataPath):
    filtered = filterAllOpenField(endoDataPath)
    res = []
    for sess in readSessions.findSessions(endoDataPath, task="openField"):
        deconv = sess.readDeconvolvedTraces(rScore=True)
        meanActivity = deconv.mean(axis=1)
        if meanActivity.isna().any(): continue
        speed = filtered.loc[str(sess)].speed
        corr = scipy.stats.pearsonr(meanActivity, speed)[0]
        shuffledCorr = scipy.stats.pearsonr(meanActivity, np.random.permutation(speed))[0]
        res.append((sess.meta.genotype, sess.meta.animal, sess.meta.date, deconv.shape[1], corr))
        res.append((sess.meta.genotype+"_shuffled", sess.meta.animal+"_shuffled",
                    sess.meta.date+"_shuffled", deconv.shape[1], shuffledCorr))
    return pd.DataFrame(res, columns=["genotype", "animal", "date", "noNeurons", "correlation"])

@cachedDataFrame("openFieldPopulationIncreaseInMovement.pkl")
def populationIncreaseInMovement(endoDataPath):
    segments = segmentAllOpenField(endoDataPath)
    res = []
    for sess in readSessions.findSessions(endoDataPath, task="openField"):
        deconv = sess.readDeconvolvedTraces(rScore=True)
        meanActivity = deconv.mean(axis=1)
        segs = segments.loc[str(sess)]
        mask = np.full(len(deconv), False)
        for r in segs[segs.behavior=='stationary'].itertuples():
            mask[r.startFrame:r.stopFrame] = True
        meanStationary = meanActivity[mask].mean()
        meanMoving = meanActivity[np.logical_not(mask)].mean()
        res.append((sess.meta.genotype, sess.meta.animal, sess.meta.date, deconv.shape[1],
                    meanStationary, meanMoving))
        
        mask = np.full(len(deconv), False)
        for r in segs[np.random.permutation(segs.behavior)=='stationary'].itertuples():
            mask[r.startFrame:r.stopFrame] = True
        meanStationary = meanActivity[mask].mean()
        meanMoving = meanActivity[np.logical_not(mask)].mean()
        res.append((sess.meta.genotype+"_shuffled", sess.meta.animal+"_shuffled", sess.meta.date+"_shuffled",
                    deconv.shape[1], meanStationary, meanMoving))
        #res.append((sess.meta.genotype+"_shuffled", sess.meta.animal+"_shuffled",
        #            sess.meta.date+"_shuffled", deconv.shape[1], shuffledCorr))
    return pd.DataFrame(res, columns=["genotype", "animal", "date", "noNeurons",
                                      "meanStationary", "meanMoving"])
#def get_centers(rois):
#    # find pixel of maximum intensity in each mask; use as neuron center
#    centers = np.array(np.unravel_index(np.array([np.argmax(roi) for roi in rois]),
#                                                  rois.shape[1:]))
#    centers = centers[::-1].T
#    return(centers)



def wAvg(group, var, weights):
    return(np.average(group[var], weights=group[weights]))
    
def bootstrap(group, var, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(group[var]), size=len(group[var]),
                               replace=True)
        avgs.append(np.average(group[var].iloc[idx], weights=group[weights].iloc[idx]))
    return(np.std(avgs))