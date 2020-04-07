import numpy as np
import pandas as pd
import seaborn as sns
import cmocean
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.svm
import sklearn.model_selection
import style
from matplotlib import animation
import pims
import itertools

from utils import readSessions, fancyViz
import analysisTunings
#from utils.cachedDataFrame import cachedDataFrame


#%% 
def getSessConfProba(dataFile, animal, date):
    selectedLabels = [phase+tt for phase in ['pL2C','mL2C','pC2L','mC2L','dL2C',
                                             'pR2C','mR2C','pC2R','mC2R','dR2C']
                               for tt in ['r.','o!']]
    def _prepareSVCTrials(deconv, lfa):
        avgSig = deconv.groupby(lfa.actionNo).mean()
        labels = lfa.groupby("actionNo").label.first()
        svcTrials = np.logical_and(avgSig.notna().all(axis=1), labels.isin(selectedLabels))
        svcX = avgSig[svcTrials]
        svcY = labels[svcTrials]
        otherTrials = np.logical_and(avgSig.notna().all(axis=1), 
                                     ~labels.isin(selectedLabels) & ~labels.str.startswith('u'))
        otherX = avgSig[otherTrials]
        otherY = labels[otherTrials]
        return svcX, svcY, otherX, otherY

    sess = next(readSessions.findSessions(dataFile, task='2choice', animal=animal, date=date))
    deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
    lfa = sess.labelFrameActions(reward="fullTrial", switch=True)
    if len(deconv) != len(lfa): return -1
    
    svcX, svcY, otherX, otherY = _prepareSVCTrials(deconv, lfa)  
    svcProbDf = []
    otherProbDf = []
    coefDf = []
    splitter = sklearn.model_selection.StratifiedKFold(5, shuffle=True, random_state=0)
    for train_idx, test_idx in splitter.split(svcX, svcY):
        trainX, trainY = svcX.iloc[train_idx,:], svcY.iloc[train_idx]
        testX, testY = svcX.iloc[test_idx,:], svcY.iloc[test_idx]
        
        svm = sklearn.svm.SVC(kernel="linear", probability=True,
                              class_weight='balanced').fit(trainX, trainY)
        
        pred = svm.predict(testX)           
        m = sklearn.metrics.confusion_matrix(testY, pred)
        m = pd.DataFrame(m, index=svm.classes_, columns=svm.classes_)
        m = m.rename_axis(index="true", columns="predicted").unstack()
        m = m.rename("occurences").reset_index()

        svcProb = svm.predict_proba(testX)
        svcProb = pd.DataFrame(svcProb, index=testX.index, columns=svm.classes_)
        svcProb['label'] = testY
        svcProbDf.append(svcProb)
        
        otherProb = svm.predict_proba(otherX)
        otherProb = pd.DataFrame(otherProb, index=otherX.index, columns=svm.classes_)
        otherProb['label'] = otherY
        otherProbDf.append(otherProb)
    
        coef = pd.DataFrame(svm.coef_, 
                            index=pd.MultiIndex.from_tuples(
                                      list(itertools.combinations(svm.classes_, 2)))
                           )
        coef = coef.loc[list(zip(selectedLabels[1::2], selectedLabels[::2]))]
        coef.index = coef.index.get_level_values(0).str.slice(0,-2)
        coef.index.name = 'phase'
        coef.columns.name = 'neuron'
        coefDf.append(coef)
        
    svcProbDf = pd.concat(svcProbDf, ignore_index=False)
    otherProbDf = pd.concat(otherProbDf, ignore_index=False)
    otherProbDf = otherProbDf.groupby(['actionNo','label']).mean().reset_index('label')
    probDf = pd.concat([svcProbDf, otherProbDf], ignore_index=False, sort=True).sort_index()
    
    coefDf = pd.concat(coefDf, ignore_index=False).groupby('phase').mean()
    coefDf /= coefDf.abs().values.max()
    
    return (m, probDf, coefDf)


#%%
confMat, probDf, coefDf = getSessConfProba('data/endoData_2019.hdf', animal='5308', date='190201')


#%% show confusion matrix
decodingData = confMat.copy()

order = [phase+tt for phase in ['pL2C','mL2C','pC2L','mC2L','dL2C',
                                'pR2C','mR2C','pC2R','mC2R','dR2C']
                  for tt in ['r.','o!']]

df = decodingData.groupby(['true','predicted']).occurences.sum().unstack()
df /= df.sum(axis=1)[:,np.newaxis]

plt.figure(figsize=(8,8))
sns.heatmap(df.reindex(order)[order], vmin=0, vmax=1, annot=True, fmt=".0%",
            cmap=cmocean.cm.amp, xticklabels=False, yticklabels=False,
            annot_kws={'fontsize': 8},  cbar=False, square=True,
            cbar_kws={'orientation':'horizontal', 'ticks':()})
plt.show()


#%%
df = probDf.set_index('label', append=True).copy()
df.columns = pd.MultiIndex.from_tuples(zip(df.columns.str.slice(0,-2),
                                           df.columns.str.slice(-2)),
                                       names=['phase','trialType'])
df = df.stack('phase')
df['phaseProb'] = df.sum(axis=1)
df['stSwProb'] = df['r.'] / (df['r.']+df['o!'])
df = df[['phaseProb','stSwProb']]

# color map for stay VS switch
colors = [style.getColor(tt) for tt in ['o!','o.','o.','r.']]
nodes = [0.,.4,.6,1.]
svcCmap = mpl.colors.LinearSegmentedColormap.from_list("svcCmap", list(zip(nodes, colors)))
sns.palplot(svcCmap(np.linspace(0,1,20)))

def getPredColor(row):
    color = np.array(svcCmap(row['stSwProb'])) # stay-switch prediction -> color coding
    color[3] = row['phaseProb']  # phase prediction -> alpha
    color = mpl.colors.to_hex(color, keep_alpha=True)
    return color

df = df.apply(getPredColor, axis=1)
df = df.unstack('phase')

# reindex to have colors per frame
sess = next(readSessions.findSessions('data/endoData_2019.hdf', task='2choice', animal='5308',
                                                                date='190201'))
lfa = sess.labelFrameActions(reward="fullTrial", switch=True).set_index(['actionNo','label'])
df = df.reindex(lfa.index).fillna(mpl.colors.to_hex((0,0,0,.75), keep_alpha=True))

df.to_pickle('cache/oprm1_5308_190201_colored_decoding.pkl')


#%%
# color map SVM weights
#colors = [style.getColor('o!'),style.getColor('o!'),(1,1,1,1),style.getColor('r.'),style.getColor('r.')]
#nodes = [0,.25,.5,.75,1.]
#weightsCmap = mpl.colors.LinearSegmentedColormap.from_list("weightsCmap", list(zip(nodes, colors)))
#sns.palplot(weightsCmap(np.linspace(0,1,20)))

def normRois(rois):
    rois /= rois.max(axis=0)
    rois = rois**1.5
    rois = (rois - .1).clip(0,.8)
    rois /= rois.max(axis=0)
    return rois

def getColoredRois(rois, colors):
    rois = np.array([rois[n].unstack('x').values for n in rois])
    if len(rois) != len(colors):
        raise ValueError("Colors must have the same length as rois.")
    rs = []
    for roi, color in zip(rois, colors):
        r = np.array([(roi > 0).astype('int')]*3) * color[:, np.newaxis, np.newaxis]
        r = np.concatenate([r, roi[np.newaxis]], axis=0)
        rs.append(r.transpose((1,2,0)))    
    rs = np.array(rs)
    return np.array(rs)

def changeAlpha(rois, fDeconv):
    rs = rois.copy()
    rs[:,:,:,3] = fDeconv[:,np.newaxis,np.newaxis] * rs[:,:,:,3]
    return rs
        
    
rois = normRois(sess.readROIs())
tunings = analysisTunings.getTuningData('data/endoData_2019.hdf')
tunings = tunings.query('animal == "5308" & date == "190201"').reset_index(drop=True).copy()
tunings['signp'] = tunings['pct'] > .995
tunings = tunings.loc[tunings.groupby('neuron').tuning.idxmax()]
colors = tunings.action.copy()
colors[~tunings.signp] = 'none'
colors = np.array([style.getColor(c[:4]) for c in colors])
rois = getColoredRois(rois, colors)

deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
deconv = (deconv-1).clip(0,14) / 14
    
    
#%%
vid = pims.PyAVReaderIndexed('data/20190201_203528_oprm1_5308-0000.avi')


#%%
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.size'] = 25
mpl.rcParams['figure.dpi'] = 90

fig, axs = plt.subplots(2, 2, figsize=(9,13),
                        gridspec_kw={'height_ratios':(.7,.3),'width_ratios':(.975,.025),
                                     'hspace':0.025,'wspace':0.15,
                                     'left':.02,'right':.9,'top':.99,'bottom':.03})


img = axs[0,0].imshow(np.zeros_like(vid[0].transpose((1,0,2))[::-1,:,:]))
axs[0,0].axis('off')

axs[0,1].axis('off')

cax = axs[1,1]
cb1 = mpl.colorbar.ColorbarBase(cmap=svcCmap, ax=cax,
                                orientation='vertical', ticks=(0,.5,1))
cb1.ax.tick_params(axis='y', length=0)
cb1.set_ticklabels((0,50,100))
cb1.set_label('P(win-stay)')
cb1.ax.yaxis.set_label_position('left')
cb1.outline.set_visible(False)

def update(frame):
    img.set_array(vid[frame].transpose((1,0,2))[::-1,:,:])
    plt.sca(axs[1,0])
    plt.cla()
    fancyViz.drawBinnedSchematicPlot(df.iloc[frame])
    return [img]

ani = animation.FuncAnimation(fig, update, frames=np.arange(10400,12800), blit=True)

writer = animation.FFMpegWriter(fps=10, bitrate=-1)
ani.save("behav_decoding.mp4", writer=writer)


#%%

fig = plt.figure()
ax = plt.gca()

imgs = [ax.imshow(np.zeros_like(r)) for r in rois]
ax.axis('off')

def update(frame):
    rs = changeAlpha(rois, deconv.loc[frame])
    [imgs[n].set_array(r) for n,r in enumerate(rs)]
    return imgs

ani = animation.FuncAnimation(fig, update, frames=np.arange(10400,11600,2), blit=True)

writer = animation.FFMpegWriter(fps=10, bitrate=-1)
ani.save("ca_vid.mp4", writer=writer)

