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
from mpl_toolkits.mplot3d import Axes3D
import style
from utils import readSessions
from itertools import product
from sklearn.decomposition import FactorAnalysis, PCA, FastICA, SparsePCA
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE

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
    deconv = s.readDeconvolvedTraces(rScore=True).reset_index(drop=True).fillna(0)

    labels = s.labelFrameActions(reward='fullTrial', switch=True, splitCenter=True)
    if len(deconv) != len(labels): continue

    # trials start upon outcome delivery
    labels['trialNo'] = (labels.label.str.contains('p[LR]2C[or][\.\!]$') * \
                         (labels.actionNo.diff() == 1).astype('int')).cumsum()
    labels = labels.set_index('trialNo')
    labels['trialType'] = labels.groupby('trialNo').label.first()
    labels['bin'] = labels.actionProgress // (1/4)
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
    deconv.columns.name = 'neuron'
    deconv = deconv.set_index(labels.index).sort_index()
    X = deconv.loc[True].groupby(['trialType','trialNo','actionNo','bin']).mean()
    X['bin'] = X.groupby(['trialType','trialNo']).cumcount()
    X = X.reset_index(['actionNo','bin'], drop=True).set_index('bin', append=True)
    X = X.loc[['pL2Cr.','pL2Co.','pL2Co!',
               'pR2Cr.','pR2Co.','pR2Co!']]
    X = X.unstack('bin')

    X['trialNo'] = X.groupby('trialType').cumcount()
    X = X.reset_index('trialNo', drop=True).set_index('trialNo', append=True)
    #X['trialNo'] = (X.groupby('trialType', as_index=False).apply(
    #                    lambda g: pd.Series(np.random.permutation(len(g)), index=g.index))
    #                    .reset_index(0, drop=True))
    #X = X.reset_index('trialNo', drop=True).set_index('trialNo', append=True)
    #X = X.stack('bin')
    X = X.stack('neuron')
    
    for k,v in [('genotype',s.meta.genotype), ('animal',s.meta.animal), ('date',s.meta.date)]:
        X.insert(0,k,v)
    X = X.set_index(['genotype','animal','date'], append=True)#.unstack(['genotype','animal','date'])
    #X = X.reorder_levels([1,2,3,0], axis=1)
    #X = X.reorder_levels([0,-1,-2,-3,2,1], axis=0)
    X = X.reorder_levels([3,4,5,2,0,1])
    
    Xs.append(X)

X = pd.concat(Xs, axis=0)

X.to_pickle('trajectory.pkl')


#%%
X = pd.read_pickle('trajectory.pkl')

#%%
def prepTrajectoryData(X, trials=None, shuffle=True, seed=None):
    T = X.copy()
    # drop trials with NaN bins
    T = T.dropna()
    # shuffle trial numbers / neuron
    if seed:
        np.random.seed(seed)
    if shuffle:
        T['trialNo'] = (T.groupby(['trialType','genotype','animal','date','neuron'], as_index=False)
                         .apply(lambda g: pd.Series(np.random.permutation(len(g)), index=g.index))
                         .reset_index(0, drop=True))
        T = T.reset_index('trialNo', drop=True).set_index('trialNo', append=True)
    # reformat, reduce to "complete" trials
    T = T.unstack(['genotype','animal','date','neuron']).stack('bin')
    T = T.loc[:,(T != np.inf).all()].dropna().copy()
    # reduce to x trials per condition
    if trials:
        T = T.query('trialNo < @trials').copy()
    # normalize matrix
    T -= T.mean(axis=1).values[:,np.newaxis]
    T /= T.std(axis=1).values[:,np.newaxis]
    return T


#%%
def plot3D(train_fits, test_fits, azimuth, angle, lims=None, bins=4,
           order=['r.','o.','o!']):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    for (tt,it,tn), df in test_fits.groupby(['trialType','iteration','trialNo']):
        ax.plot(df[0], df[1], df[2], c=style.getColor(tt[-2:]),
                alpha=.08, lw=.8, zorder={'r.':3, 'o.':2, 'o!':1}[tt[-2:]])
    
    if not lims: 
        ax.set_xlim(np.array(ax.get_xlim())*1.3)
        ax.set_ylim(np.array(ax.get_ylim())*1.3)
        ax.set_zlim(np.array(ax.get_zlim())*1.3)
    else: 
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_zlim(lims)
#    lims = (ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0])
    
    meandf = train_fits.groupby(['trialType','bin']).mean().reset_index()
    for tt, df in meandf.groupby('trialType'):
        ax.plot(df[0], df[1], df[2], c=style.getColor(tt[-2:]),
                alpha=.6, lw=1.5, zorder={'r.':3, 'o.':2, 'o!':1}[tt[-2:]])
#        ax.plot(df[0], df[1], lims[2], zdir='z', c=style.getColor(tt[-2:]),
#                alpha=.6, lw=1.25)
#        ax.plot(df[0], df[2], lims[1], zdir='y', c=style.getColor(tt[-2:]),
#                alpha=.6, lw=1.25)
#        ax.plot(df[1], df[2], lims[0], zdir='x', c=style.getColor(tt[-2:]),
#                alpha=.6, lw=1.25)
        ax.scatter(df[0], df[1], df[2], color=style.getColor(tt[-2:]),
                   alpha=.6, s=15, lw=0, zorder={'r.':3, 'o.':2, 'o!':1}[tt[-2:]])
#        ax.scatter(df[0], df[1], lims[2], zdir='z', color=style.getColor(tt[-2:]),
#                   alpha=.6, s=10, lw=0)
#        ax.scatter(df[0], df[2], lims[1], zdir='y', color=style.getColor(tt[-2:]),
#                   alpha=.6, s=10, lw=0)
#        ax.scatter(df[1], df[2], lims[0], zdir='x', color=style.getColor(tt[-2:]),
#                   alpha=.6, s=10, lw=0)
        df = df.loc[df.bin % bins == 0]
        ax.scatter(df[0], df[1], df[2], color=style.getColor(tt[-2:]),
                   alpha=.8, s=50, lw=0, zorder={'r.':3, 'o.':2, 'o!':1}[tt[-2:]])
#        ax.scatter(df[0], df[1], lims[2], zdir='z', color=style.getColor(tt[-2:]),
#                   alpha=.6, s=30, lw=0)
#        ax.scatter(df[0], df[2], lims[1], zdir='y', color=style.getColor(tt[-2:]),
#                   alpha=.6, s=30, lw=0)
#        ax.scatter(df[1], df[2], lims[0], zdir='x', color=style.getColor(tt[-2:]),
#                   alpha=.6, s=30, lw=0)
    
    ax.view_init(azimuth, angle)
    #ax.set_title('{} {}'.format(azimuth, angle))
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_zticklabels(())
    
    return fig


#%%
T = prepTrajectoryData(X.query('trialType in ["pR2Cr.","pR2Co.","pR2Co!"]'),
                       shuffle=True, seed=0, trials=999999)

pca = PCA(3, whiten=True, svd_solver='full')
pca.fit(T)

train_fits = pd.DataFrame(pca.transform(T), index=T.index).reset_index()

Ts = []
for i in range(5):
    Ts.append(prepTrajectoryData(X, shuffle=True, seed=i+1, trials=20))
T = pd.concat(Ts, keys=np.arange(len(Ts)), names=['iteration'])

test_fits = pd.DataFrame(pca.transform(T), index=T.index).reset_index()

#%%
fig = plot3D(train_fits, test_fits.loc[test_fits.trialType.str.slice(0,4) == 'pR2C'],
             290, 45, lims=(-3,3))
fig.savefig('svg/right_trials_290_45.png', pad_inches=0, bbox_inches='tight')

fig = plot3D(train_fits,
             test_fits.loc[test_fits.trialType.str.slice(0,4) == 'pR2C'],
             195, 180, lims=(-3,3))
fig.savefig('svg/right_trials_195_180.png', pad_inches=0, bbox_inches='tight')


#%% more BS ########################################################################
T = prepTrajectoryData(X.loc[['oprm1']], shuffle=True, trials=9999)
T = T.groupby(['trialType','bin']).mean()
pca = PCA(25, whiten=True, svd_solver='full')
tsne = TSNE(3, method='exact', init='pca', n_iter=10000)
pca_trans = pca.fit_transform(T)
tsne_trans = tsne.fit_transform(pca_trans)
fit = pd.DataFrame(tsne_trans, index=T.index)

#%% try single session; can project real trials into space obtained from shuffled data
genotype = 'oprm1'
pca_dims = 10 #10
pca = PCA(whiten=True, svd_solver='full') # 10
iso = Isomap(5,3,max_iter=10000,eigen_solver='dense',n_jobs=-1,path_method='D',
             neighbors_algorithm='brute')
##%%
Ts = []
for i in range(1):
    T = prepTrajectoryData(X.query('genotype == @genotype' + \
                                   ' and trialType in ["pL2Co.","pL2Cr."]'),
                           shuffle=True, trials=20, seed=i+10)
    Ts.append(T)
T = pd.concat(Ts, keys=np.arange(len(Ts)), names=['iteration'])

pca.fit(T)
pca_trans = pca.transform(T)
iso.fit(pca_trans[:,:pca_dims])
iso_trans = iso.transform(pca_trans[:,:pca_dims])

main_fits = pd.DataFrame(iso_trans, index=T.index).reset_index()


#%%
Ts = []
for i in range(1):
    Ts.append(prepTrajectoryData(X.loc[[genotype]], shuffle=False, trials=20, seed=i+10*200))
T = pd.concat(Ts, keys=np.arange(len(Ts)), names=['iteration'])

pca_trans = pca.transform(T)
iso_trans = iso.transform(pca_trans[:,:pca_dims])
fits = pd.DataFrame(iso_trans, index=T.index).reset_index()


#%%
def plotSide3D(sides, azimouth, angle, bins=3, phases=['o!','o.','r.']):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    for n,side in enumerate(sides):
        if 'o!' in phases:
            #df = mfit.query('trialType == "p{}2Co!"'.format(side))
            #ax.plot(df[0], df[1], df[2], 
            #        c=sns.desaturate(style.getColor('o!'), [1,1][n]),
            #        alpha=[.7,.7][n], lw=3)
            #df = df.loc[df.bin % bins == 0]
            #ax.scatter(df[0], df[1], df[2], color=style.getColor('o!'),
            #           s=50, depthshade=False, alpha=[.8,.8][n],
            #           marker='<' if side == 'L' else '>')
            #c=df.bin, cmap='rainbow', s=30)
            #for _, df in fit.query('trialType == "p{}2Co!"'.format(side)).groupby('trialNo'):
            #    ax.plot(df[0], df[1], df[2], c=style.getColor('o!'), alpha=[.15,.15][n],
            #            lw=1)
            for _, df in mfits.query('trialType=="p{}2Co!"'.format(side)).groupby('iteration'):
                ax.plot(df[0], df[1], df[2], c=style.getColor('o!'), alpha=[.25,.25][n],
                        lw=1)
        
        if 'o.' in phases:
            #df = mfit.query('trialType == "p{}2Co."'.format(side))
            #ax.plot(df[0], df[1], df[2],
            #        c=sns.desaturate(style.getColor('o.'), [1,1][n]),
            #        alpha=[.7,.7][n], lw=3)
            #df = df.loc[df.bin % bins == 0]
            #ax.scatter(df[0], df[1], df[2], color=style.getColor('o.'),
            #           s=50, depthshade=False, alpha=[.8,.8][n],
            #           marker='<' if side == 'L' else '>')
            #c=df.bin, cmap='rainbow', s=30)
            #for _, df in fit.query('trialType == "p{}2Co."'.format(side)).groupby('trialNo'):
            #    ax.plot(df[0], df[1], df[2], c=style.getColor('o.'), alpha=[.15,.15][n],
            #            lw=1)
            for _, df in mfits.query('trialType=="p{}2Co."'.format(side)).groupby('iteration'):
                ax.plot(df[0], df[1], df[2], c=style.getColor('o.'), alpha=[.25,.25][n],
                        lw=1)

        if 'r.' in phases:
            #df = mfit.query('trialType == "p{}2Cr."'.format(side))
            #ax.plot(df[0], df[1], df[2],
            #        c=sns.desaturate(style.getColor('r.'), [1,1][n]),
            #        alpha=[.7,.7][n], lw=3)
            #df = df.loc[df.bin % bins == 0]
            #ax.scatter(df[0], df[1], df[2], color=style.getColor('r.'),
            #           s=50, depthshade=False, alpha=[.8,.8][n],
            #           marker='<' if side == 'L' else '>')
            #c=df.bin, cmap='rainbow', s=30)
            #for _, df in fit.query('trialType == "p{}2Cr."'.format(side)).groupby('trialNo'):
            #    ax.plot(df[0], df[1], df[2], c=style.getColor('r.'), alpha=[.15,.15][n],
            #            lw=1)
            for _, df in mfits.query('trialType=="p{}2Cr."'.format(side)).groupby('iteration'):
                ax.plot(df[0], df[1], df[2], c=style.getColor('r.'), alpha=[.25,.25][n],
                        lw=1)
    
    #ax.axis('off')
    ax.view_init(azimouth, angle)
    ax.set_title('{} {}'.format(azimouth, angle))
    #plt.show()

    
plotSide3D('LR', 130, 35, phases=['r.','o.','o!'])
#plt.axis('off')
plt.show()

#%%
fig = plt.figure()
ax = plt.gca()
x,y = 1,2

df = mfit.query('trialType == "pL2Co!"')
ax.plot(df[x], df[y], c=style.getColor('o!'), alpha=.7, lw=2.5)
df = df.loc[df.bin % 4 == 0]
ax.scatter(df[x], df[y], c='k', s=30, marker='<')#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Co!"').groupby('trialNo'):
    ax.plot(df[x], df[y], c=style.getColor('o!'), alpha=.15)

df = mfit.query('trialType == "pL2Co."')
ax.plot(df[x], df[y], c=style.getColor('o.'),alpha=.7, lw=2.5)
df = df.loc[df.bin % 4 == 0]
ax.scatter(df[x], df[y], c='k', s=30, marker='<')#c=df.bin, cmap='rainbow', s=30)
#for _, df in fit.query('trialType == "pL2Co."').groupby('trialNo'):
#    ax.plot(df[x], df[y], c=style.getColor('o.'), alpha=.15)

df = mfit.query('trialType == "pL2Cr."')
ax.plot(df[x], df[y], c=style.getColor('r.'), alpha=.7, lw=2.5)
df = df.loc[df.bin % 4 == 0]
ax.scatter(df[x], df[y], c='k', s=30, marker='<')#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Cr."').groupby('trialNo'):
    ax.plot(df[x], df[y], c=style.getColor('r.'), alpha=.15)

#plt.show()

#%%
#fig = plt.figure()
#ax = plt.gca()
#x,y = 1,2

df = mfit.query('trialType == "pR2Co!"')
ax.plot(df[x], df[y], c=style.getColor('o!'),alpha=.7, lw=2.5)
df = df.loc[df.bin % 4 == 0]
ax.scatter(df[x], df[y], c='k', s=30, marker='>')#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Co!"').groupby('trialNo'):
    ax.plot(df[x], df[y], c=style.getColor('o!'), alpha=.15)

df = mfit.query('trialType == "pR2Co."')
ax.plot(df[x], df[y], c=style.getColor('o.') ,alpha=.7, lw=2.5)
df = df.loc[df.bin % 4 == 0]
ax.scatter(df[x], df[y], c='k', s=30, marker='>')#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Co."').groupby('trialNo'):
    ax.plot(df[x], df[y], c=style.getColor('o.'), alpha=.15)

df = mfit.query('trialType == "pR2Cr."')
ax.plot(df[x], df[y], c=style.getColor('r.'), alpha=.7, lw=2.5)
df = df.loc[df.bin % 4 == 0]
ax.scatter(df[x], df[y], c='k', s=30, marker='>')#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Cr."').groupby('trialNo'):
    ax.plot(df[x], df[y], c=style.getColor('r.'), alpha=.15)
    

#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = mfit.query('trialType == "pL2Co!"')
ax.plot(df[0], df[1], df[2], c='r',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
#for _, df in fit.query('trialType == "pL2Co!"').groupby('trialNo'):
#    ax.plot(df[0], df[1], df[2], c='r', alpha=.1)

df = mfit.query('trialType == "pL2Co."')
ax.plot(df[0], df[1], df[2], c='y',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
#for _, df in fit.query('trialType == "pL2Co."').groupby('trialNo'):
#    ax.plot(df[0], df[1], df[2], c='y', alpha=.1)

df = mfit.query('trialType == "pL2Cr."')
ax.plot(df[0], df[1], df[2], c='g', alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
#for _, df in fit.query('trialType == "pL2Cr."').groupby('trialNo'):
#    ax.plot(df[0], df[1], df[2], c='g', alpha=.1)

plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = mfit.query('trialType == "pL2Co!"')
ax.plot(df[0], df[1], df[2], c='r',alpha=.9, lw=2.5, ls=':')
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Co!"').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='r', alpha=.1)

df = mfit.query('trialType == "pL2Co."')
ax.plot(df[0], df[1], df[2], c='y',alpha=.9, lw=2.5, ls=':')
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Co."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='y', alpha=.1)

df = mfit.query('trialType == "pL2Cr."')
ax.plot(df[0], df[1], df[2], c='g', alpha=.9, lw=2.5, ls=':')
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pL2Cr."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='g', alpha=.1)

df = mfit.query('trialType == "pR2Co!"')
ax.plot(df[0], df[1], df[2], c='r',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Co!"').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='r', alpha=.1)

df = mfit.query('trialType == "pR2Co."')
ax.plot(df[0], df[1], df[2], c='y',alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Co."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='y', alpha=.1)

df = mfit.query('trialType == "pR2Cr."')
ax.plot(df[0], df[1], df[2], c='g', alpha=.9, lw=2.5)
df = df.loc[df.bin % 3 == 0]
ax.scatter(df[0], df[1], df[2], c='k', s=30, depthshade=False)#c=df.bin, cmap='rainbow', s=30)
for _, df in fit.query('trialType == "pR2Cr."').groupby('trialNo'):
    ax.plot(df[0], df[1], df[2], c='g', alpha=.1)

plt.show()