#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:33:12 2019

@author: mowe
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import sklearn.neighbors
import tqdm

from utils import readSessions

def getSignalHistogram(endoDataPath, task="2choice"):
    bins = np.arange(-5, 3., 0.05)
    hist = np.zeros(len(bins)-1)
    nZero = 0
    nNonZero = 0
    for sess in tqdm.tqdm(readSessions.findSessions(endoDataPath, task=task), total=66):
        deconv = sess.readDeconvolvedTraces(zScore=False)
        deconv = deconv[deconv.notnull().any(axis=1)]
        deconv /= deconv.std()
        mask = deconv.values.flat > 0
        nNonZero += mask.sum()
        nZero += len(mask) - mask.sum()
        vals = deconv.values.flat[mask]
        hist += np.histogram(np.log10(vals), bins)[0]
    s = np.concatenate(([nZero], hist))
    i = np.concatenate(([0], 10**(0.5*(bins[:-1] + bins[1:]))))
    return pd.Series(s, index=i)

def getVarianceExplainedByPCA(endoDataPath, task="2choice"):
    dfs = []
    for sess in tqdm.tqdm(readSessions.findSessions(endoDataPath, task=task), total=66):
        deconv = sess.readDeconvolvedTraces(zScore=True)
        deconv = deconv[deconv.notnull().any(axis=1)]
        for sigma in (0, 1, 5, 10, 20):
            if sigma == 0:
                signal = deconv.values.copy()
            else:
                signal = scipy.ndimage.gaussian_filter1d(deconv, sigma, axis=0)

            pca = sklearn.decomposition.PCA().fit(signal)
            dfs.append(pd.DataFrame({'session': str(sess), 'sigma': sigma, 'totNeurons': signal.shape[1], 
                                     'numNeuronsIncluded': np.arange(1, signal.shape[1]+1),
                                     'fracVarExplained': np.cumsum(pca.explained_variance_ratio_)}))
    return pd.concat(dfs)

def dimPerTime(values, nNeighbours = 500, dropClosest = 3, desc = ""):
    ballTree = sklearn.neighbors.BallTree(values)
    dists, ids = ballTree.query(values, nNeighbours)
    N = len(values)
    dim = np.zeros(N)
    EPS = 1e-8
    for i in tqdm.trange(N, desc=desc):
        X = np.vstack((np.ones(nNeighbours), np.log(dists[i]+EPS))).T[dropClosest:]
        Y = np.log(np.arange(dropClosest,nNeighbours))
        dim[i] = np.linalg.lstsq(X, Y, rcond=None)[0][1]
    return dim

def calculateTopologicalDimensionality(endoDataPath, task="2choice"):
    dfs = []
    for sess in readSessions.findSessions(endoDataPath, task=task):
        deconv = sess.readDeconvolvedTraces(zScore=True)
        if deconv.isna().any().any(): continue
        dpt = dimPerTime(deconv, desc=str(sess))
        dfs.append(pd.DataFrame({'session': str(sess), 'nNeurons': deconv.shape[1],
                                 'nNeighbours': 500,  'time': deconv.index, 'dimensionality': dpt}))
    return pd.concat(dfs)
    
#%%
def getKMeansScores(tunings):
    #def shuffleTunings(tunings): # shuffle each column (by action) to create fake tunings
    #    shuffled = tunings.copy()
    #    shuffled.apply(np.random.shuffle, axis=0) # in place
    #    return shuffled
    
    def getSilhouetteScore(tunings, n_clusters, random_state):
        labels = (KMeans(n_clusters=n_clusters, init='random', random_state=random_state,
                         n_init=1, max_iter=1000, n_jobs=-1)
                        .fit(tunings).labels_)
        return silhouette_score(tunings, labels=labels)
    
    
    score_df = pd.DataFrame()
    for gt, gtTunings in tunings.groupby('genotype'):
        for n in tqdm.trange(2,51,desc=gt): # loop through range of #clusters
            scores_real = []
            #scores_shuffle = []
            # get 1000 clusterings and associated silhouette scores
            for i in range(1000):
                #gtShuffle = shuffleTunings(gtTunings)
                scores_real.append(getSilhouetteScore(gtTunings, n, i))
                #scores_shuffle.append(getSilhouetteScore(gtShuffle, n, i))
            # store mean and standard deviation of silhouette scores
            score_df = score_df.append(pd.Series({'score_avg':np.mean(scores_real),
                                                  'score_std':np.std(scores_real),
                                                  #'score_shuffle_avg':np.mean(scores_shuffle),
                                                  #'score_shuffle_std':np.std(scores_shuffle),
                                                  'genotype':gt,
                                                  'n_clusters':n}),
                                       ignore_index=True)

