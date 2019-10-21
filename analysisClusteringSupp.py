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
import tqdm


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

