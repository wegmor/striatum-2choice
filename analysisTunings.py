#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:24:45 2019

@author: mowe
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from utils import readSessions, windowUtils
from utils.cachedDataFrame import cachedDataFrame

#%%
#def get_centers(rois):
#    # find pixel of maximum intensity in each mask; use as neuron center
#    centers = np.array(np.unravel_index(np.array([np.argmax(roi) for roi in rois]),
#                                                  rois.shape[1:]))
#    centers = centers[::-1].T
#    return(centers)
    
def getActionAverages(traces, apf):
    keepLabels = ['pC2L-', 'mC2L-',
                  'pC2R-', 'mC2R-',
                  'dL2C-', 'pL2Co', 'pL2Cr', 'mL2C-',
                  'dR2C-', 'pR2Co', 'pR2Cr', 'mR2C-']
    apf = apf.loc[apf.label.isin(keepLabels)].copy()
    apf['label'] = apf.label.astype('str')
    actionAvg = traces.loc[apf.index].groupby([apf.label,apf.actionNo]).mean().dropna()
    labels = apf.groupby(['label','actionNo']).first().loc[actionAvg.index,
                                                           ['actionDuration']]    
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
    
    
#%%
def _getTuningData(dataFilePath, no_shuffles=1000, on_shuffled=False):
    df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='2choice'):
        traces = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True) # frame no as index
        if on_shuffled:
            apf = s.shuffleFrameLabels(switch=False, reward='sidePorts',
                                       splitCenter=True)[0].reset_index(drop=True)
        else:
            apf = s.labelFrameActions(switch=False, reward='sidePorts',
                                      splitCenter=True).reset_index(drop=True)
        
        # TODO: fix remaining recordings with dropped frames
        if traces.shape[0] != apf.shape[0]:
            continue
        
        actionAvg, labels = getActionAverages(traces, apf)  # mean per action
        actionAvg = actionAvg.groupby('label').mean()       # mean per label

        shuffle_labels = s.shuffleFrameLabels(no_shuffles, switch=False,
                                              reward='sidePorts', splitCenter=True)
        
        s_actionAvgs = []
        for s_apf in shuffle_labels:
            s_actionAvg, s_labels = getActionAverages(traces, s_apf)
            s_actionAvg = s_actionAvg.groupby('label').mean()
            s_actionAvgs.append(s_actionAvg)
        s_actionAvgs = pd.concat(s_actionAvgs, keys=np.arange(no_shuffles),
                                 names=['shuffle']).reorder_levels(['label','shuffle'])

        for action, adata in s_actionAvgs.groupby('label'):
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

@cachedDataFrame("actionTunings.pkl")
def getTuningData(dataFilePath, no_shuffles=1000):
    df = _getTuningData(dataFilePath, no_shuffles, on_shuffled=False)
    return df

@cachedDataFrame("actionTunings_shuffled.pkl")
def getTuningData_shuffled(dataFilePath, no_shuffles=1000):
    df = _getTuningData(dataFilePath, no_shuffles, on_shuffled=True)
    return df


#%%
def getTunedNoHistData(tuningData):
    count_df = (tuningData.groupby(['genotype','animal','date','neuron'])[['signp']]
                          .sum().astype('int').copy())
    
    hist_df = pd.DataFrame()
    for (g,a,d), data in count_df.groupby(['genotype','animal','date']):
        signpHist = pd.Series(dict(zip(np.arange(13), 
                                       np.bincount(data.signp, minlength=13))))
        df = pd.DataFrame({'signp':signpHist})
        df.index.name = 'count'
        df['genotype'], df['animal'], df['date'] = g, a, d
        hist_df = hist_df.append(df.reset_index().set_index(['genotype','animal','date','count']))
    
    hist_df = hist_df.reset_index('count')
    hist_df['bin'] = pd.cut(hist_df['count'], bins=[-.5,.5,1.5,2.5,3.5,4.5,13]).cat.codes
    hist_df = (hist_df.groupby(['genotype','animal','date','bin'])[['signp']].sum()
                      .reset_index('bin'))
    hist_df['noNeurons'] = count_df.groupby(['genotype','animal','date']).size()
    hist_df['signp'] /= hist_df.noNeurons
    
    return hist_df


#%% TSNE
@cachedDataFrame("tuning_tsne.pkl")
def getTSNEProjection(tuningData, perplexity=30):
    df = tuningData.set_index(['genotype','animal','date','neuron']).copy()
    df["signp"] = df.pct > 0.995
    df = df.loc[df.groupby(['genotype','animal','date','neuron']).signp.sum() >= 1]
    df = df.set_index('action', append=True).tuning.unstack('action')
    
    tsne = TSNE(perplexity=perplexity, n_iter=10000, init='pca').fit_transform(df)
    tsne_df = pd.DataFrame(tsne, index=df.index).reset_index()
    
    df = tuningData.copy()
    df["signp"] = df.pct > 0.995
    maxdf = df.loc[df.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
    maxdf = maxdf.loc[df.signp]
    #maxdf.loc[~df.signp, 'action'] = 'none'
    tsne_df = tsne_df.merge(maxdf[['genotype','animal','date','neuron','action','tuning']],
                            on=['genotype','animal','date','neuron'])
    
    return tsne_df


#%% similarly tuned neurons closer than chance?
@cachedDataFrame("tuning_pdists.pkl")
def getPDistData(dataFilePath, tuningData, no_shuffles=1000):
    dist_df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='2choice'):
        # load ROI centers
        roics = np.array(s.readROIs().idxmax(axis=0).tolist())
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


#%%
@cachedDataFrame('distVsCorrData.pkl')
def getPDistVsCorrData(dataFilePath):
    dist_cc_df = pd.DataFrame()
    
    for s in readSessions.findSessions(dataFilePath, task=['openField','2choice']): #, task='2choice'):    
        deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True).iloc[:30*60*20] # first 30 min
        # drop everything but top 20% biggest events
        #pct80 = np.nanpercentile(deconv[deconv != 0], 80, axis=0)
        #deconv = deconv[deconv >= pct80].fillna(0)
        
        coords = np.array(s.readROIs().idxmax(axis=0).tolist())
        coords_shuffled = np.random.permutation(coords)
    
        ##%%
        # calculate pairwise pearson correlations (downsample to 200ms bins)
        pcc = deconv.rolling(4).mean()[3::4].corr('pearson').values
        # calculate pairwise distances between original positions
        pdist_orig = squareform(pdist(coords))
        # inscopix says 1440 px -> 900 um; 4x downsampled 900/360 = 2.5
        pdist_orig = pdist_orig * 2.5
        # calculate pairwise distances between shuffeled positions (i.e. shuffled
        # distances)
        pdist_perm = squareform(pdist(coords_shuffled))
        pdist_perm = pdist_perm * 2.5
        
        # merge into dataframe (correlation, orig distance, shuffled distance)
        n_neurons = len(coords)
        df = pd.DataFrame({'cc': pcc[np.triu_indices(n_neurons,1)],
                           'dist_orig': pdist_orig[np.triu_indices(n_neurons,1)],
                           'dist_perm': pdist_perm[np.triu_indices(n_neurons,1)],
                           'noNeurons': n_neurons,
                           'genotype': s.meta.genotype,
                           'animal': s.meta.animal,
                           'date': s.meta.date,
                           'task': s.meta.task}) 
        dist_cc_df = dist_cc_df.append(df)
        
    return dist_cc_df


#%%
def getTaskNoTaskData(dataFilePath):
    df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='2choice'):
        deconv = s.readDeconvolvedTraces(rScore=True) #, indicateBlocks=True)
        D = deconv.mean(axis=1)
        L = s.labelFrameActions()
        #T = s.readTracking(inCm=True)
        
        if len(D) != len(L): continue
    
#        T.index = D.index.copy()
#        T = T.iloc[:30*60*20]
#        T.index = T.index.remove_unused_levels()
#        
#        filtered = []
#        for block in T.index.levels[0]:
#            t = T.loc[block]
#            filtered.append(particleFilter.particleFilter(t, nParticles=2000))
#        F = pd.concat(filtered).reset_index(drop=True)
    
        D = D.reset_index(drop=True).iloc[:30*60*20]
        L = L.iloc[:30*60*20]
        
        eD = D.loc[~L.label.str.startswith('u')].mean()
        #eF = F.loc[~L.label.str.startswith('u'), 'speed'].mean() * 20
        uD = D.loc[L.label.str.startswith('u')].mean()
        #uF = F.loc[L.label.str.startswith('u'), 'speed'].mean() * 20
        
        df = df.append({'engaged': eD,
                        'unengaged': uD,
                        #'engagedSpeed': eF,
                        #'unengagedSpeed': uF,
                        'task': s.meta.task,
                        'genotype': s.meta.genotype,
                        'animal': s.meta.animal,
                        'date': s.meta.date,
                        'noNeurons': deconv.shape[1]},
                       ignore_index=True)
        
    return df

@cachedDataFrame('phaseRasterData.pkl')
def getPhaseRasterData(dataFile):
    tuningData = getTuningData().set_index(["action", "animal", "date", "neuron"])
    allTunedNeurons = (tuningData.pct > 0.995).sort_index()
    res = []
    for sess in readSessions.findSessions(dataFile, task="2choice"):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        lfa = sess.labelFrameActions(reward="sidePorts")
        if len(deconv) != len(lfa):
            continue
        for action in ("mC2L-", "mC2R-", "mL2C-", "mR2C-"):
            key = (action, sess.meta.animal, sess.meta.date)
            tunedNeurons = allTunedNeurons.loc[key]
            if not tunedNeurons.any():
                continue
            bins = windowUtils.binAroundAction(lfa, action).bin
            raster = deconv.loc[:, tunedNeurons].groupby(bins).mean().T
            raster["action"] = action
            raster["genotype"] = sess.meta.genotype
            raster["animal"] = sess.meta.animal
            raster["date"] = sess.meta.date
            raster = raster.set_index(["action", "genotype", "animal", "date", raster.index.rename("neuron")])
            res.append(raster)
    return pd.concat(res, axis=0).sort_index()
