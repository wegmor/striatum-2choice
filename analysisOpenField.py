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

from utils import readSessions, particleFilter, segmentBehaviors

def segmentAllOpenField(dataFile):
    allBehaviors = []
    for sess in readSessions.findSessions(dataFile, task="openField"):
        tracking = sess.readTracking(inCm=True)
        coords = particleFilter.particleFilter(tracking, flattening = 1e-12)
        coords.rename(columns={"bodyAngle": "bodyDirection"}, inplace=True)
        coords.rename_axis("time", axis=0, inplace=True)
        coords.bodyDirection *= 180/np.pi
        behaviors = segmentBehaviors.segmentBehaviors(coords)
        behaviors.insert(0, "session", str(sess))
        behaviors.insert(1, "actionNo", behaviors.index.copy())
        allBehaviors.append(behaviors)
    return pd.concat(allBehaviors)

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
    labels = behaviors.groupby("actionNo").behavior.first()
    duration = behaviors.groupby("actionNo").size()
    validTrials = np.logical_and(avgSig.notna().all(axis=1), duration >= minDuration)
    X = avgSig[validTrials]
    Y = labels[validTrials]
    return X, Y

def decodeWithIncreasingNumberOfNeurons(dataFile, allBehaviors):
    nShufflesPerNeuronNum = 20
    with multiprocessing.Pool(5) as pool:
        res = []
        for sess in readSessions.findSessions(dataFile, task="openField"):
            deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
            behaviors = allBehaviors.loc[str(sess)].set_index("startFrame", drop=True)[["actionNo", "behavior"]]
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

def decodingConfusion(dataFile, allBehaviors):
    confMats = []
    for sess in readSessions.findSessions(dataFile, task="openField"):
        deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
        behaviors = allBehaviors.loc[str(sess)].set_index("startFrame", drop=True)[["actionNo", "behavior"]]
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
def getTuningData(dataFilePath, allBehaviors, no_shuffles=1000):
    df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='openField'):
        traces = s.readDeconvolvedTraces(zScore=True).reset_index(drop=True) # frame no as index
        behaviors = allBehaviors.loc[str(s)].set_index("startFrame", drop=True)[["actionNo", "behavior"]]
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

        deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
        
        #Select good frames to use
        mask = minWallDist < 5 #Head closer than 5 cm to the wall
        mask[deconv.isna().any(axis=1)] = False #Discard frames where neural data is missing
        mask[likelihood<0.9] = False #Discard frames where DeepLabCut is uncertain
        
        #Setup in the format expected by scikit learn
        X = deconv[mask]
        Y_xx = wall_xx[mask]
        Y_yy = wall_yy[mask]
        
        svr = sklearn.svm.SVR("linear")
        cv = sklearn.model_selection.KFold(5)
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