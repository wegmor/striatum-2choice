#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:57:06 2019

@author: mowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import figurefirst
import style
from utils import readSessions
from itertools import product
from sklearn.decomposition import FactorAnalysis, PCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap

plt.ioff()
style.set_context()

#%%
endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path("data") / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#%%
Xs = []
for s in readSessions.findSessions(endoDataPath, task='2choice'):
    deconv = s.readDeconvolvedTraces().reset_index(drop=True).fillna(0)
    
    # 10 min zscore window
    window = 10*60*20
    df = pd.concat([deconv.iloc[:window//2+1],deconv,deconv.iloc[-(window//2+1):]])
    df = (df - df.rolling(window, center=True).mean()) / df.rolling(window, center=True).std()
    deconv = df.iloc[window//2+1:-(window//2+1)]

    labels = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
    if len(deconv) != len(labels): continue

    # trials start upon outcome delivery
    labels['trialNo'] = (labels.label.str.contains('p[LR]2C[or][\.\!]$') * \
                         (labels.actionNo.diff() == 1).astype('int')).cumsum()
    labels = labels.set_index('trialNo')
    labels['trialType'] = labels.groupby('trialNo').label.first()
    labels['bin'] = labels.actionProgress // (1/3)
    labels = labels.reset_index().set_index(['trialType','trialNo'])
    labels = labels.sort_index()
    # only keep actions with all actions longer than 5 frames
    labels['include'] = (labels.groupby(['trialType','trialNo']).actionDuration.apply(
                             lambda t: (np.unique(t) >= 5).all()))
    # make sure trials included follow trial structure perfectly
    properTrials = [l1+l2 for l1,l2 in product((['pL2C','mL2C'],['pR2C','mR2C']),
                                               (['pC2L','mC2L','dL2C'],['pC2R','mC2R','dR2C']))]
    labels['include'] = (labels.include & (labels.groupby(['trialType','trialNo','actionNo']).label.first()
                                                 .groupby(['trialType','trialNo'])
                                                 .apply(lambda t: [l[:-2] for l in t] in properTrials)))

    labels = labels.sort_values(['actionNo','actionProgress'])
    labels = labels.reset_index().set_index(['include','trialType','trialNo','actionNo','bin'])
    
    # if there are fewer than 20 trials in any trial type, omit! -> smallest # trials determines
    # the size of the final collated data set
    if not (labels.loc[True].groupby(['trialType','trialNo']).first().groupby('trialType').size()
                  .loc[['pL2Cr.','pL2Co.','pL2Co!', 'pR2Cr.','pR2Co.','pR2Co!']] >= 20).all():
        continue

    # mean activity for bins
    deconv = deconv.set_index(labels.index).sort_index()
    X = deconv.loc[True].groupby(['trialType','trialNo','actionNo','bin']).mean()
    X['bin'] = X.groupby(['trialType','trialNo']).cumcount()
    X = X.reset_index(['actionNo','bin'], drop=True).set_index('bin', append=True)
    X = X.loc[['pL2Cr.','pL2Co.','pL2Co!',
               'pR2Cr.','pR2Co.','pR2Co!']]
    X = X.unstack('bin')
    #X['trialNo'] = X.groupby('trialType').cumcount()
    X['trialNo'] = (X.groupby('trialType', as_index=False).apply(
                        lambda g: pd.Series(np.random.permutation(len(g)), index=g.index))
                        .reset_index(0, drop=True))
    X = X.reset_index('trialNo', drop=True).set_index('trialNo', append=True)
    X = X.stack('bin')
    X.columns.name = 'neuron'
    
    for k,v in [('genotype',s.meta.genotype), ('animal',s.meta.animal), ('date',s.meta.date)]:
        X.insert(0,k,v)
    X = X.set_index(['genotype','animal','date'], append=True).unstack(['genotype','animal','date'])
    X = X.reorder_levels([1,2,3,0], axis=1)
    
    Xs.append(X)

#%%
X = pd.concat(Xs, axis=1)

#df = X.loc[:,'oprm1'].dropna().copy()
#df = X.loc[['pR2Co!','pR2Co.','pR2Cr.'],:].dropna().copy()
df = X.loc[:,(X != np.inf).all()].dropna().copy()
#df = X.loc[['pL2Co!','pL2Co.','pL2Cr.'],(X != np.inf).all()].dropna().copy()
df = df.query('bin >= 6')

#fa = LocallyLinearEmbedding(10, 3, max_iter=1000, method='hessian', eigen_solver='dense')
#fa = FastICA(3)
fa = PCA(3)
#fa = Isomap(5,3,max_iter=10000)
#fa = MDS(3, metric=True, max_iter=10000)
fit = fa.fit_transform(df)
fit = pd.DataFrame(fit, index=df.index)

mfit = fit.groupby(['trialType','bin']).mean()
mfit = mfit.reset_index()
fit = fit.reset_index()

#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = mfit.query('trialType == "pR2Co!"')
ax.plot(df[0], df[1], df[2], c='r',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Co!"').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='r', alpha=.25)

df = mfit.query('trialType == "pR2Co."')
ax.plot(df[0], df[1], df[2], c='y',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Co."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='y', alpha=.25)

df = mfit.query('trialType == "pR2Cr."')
ax.plot(df[0], df[1], df[2], c='g', alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Cr."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='g', alpha=.25)

plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = mfit.query('trialType == "pL2Co!"')
ax.plot(df[0], df[1], df[2], c='r',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Co!"').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='r', alpha=.1)

df = mfit.query('trialType == "pL2Co."')
ax.plot(df[0], df[1], df[2], c='y',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Co."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='y', alpha=.1)

df = mfit.query('trialType == "pL2Cr."')
ax.plot(df[0], df[1], df[2], c='g', alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Cr."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='g', alpha=.1)

plt.show()