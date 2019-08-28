import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pathlib
import figurefirst
import cmocean

#import sys
#thisFolder = pathlib.Path(__file__).resolve().parent
#sys.path.append(str(thisFolder.parent))

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisDecoding
import style

style.set_context()
plt.ioff()


#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path('svg')
cacheFolder = pathlib.Path('cache')
templateFolder = pathlib.Path('templates')

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()
    
    
#%%
layout = figurefirst.FigureLayout(templateFolder / "decoding.svg")
layout.make_mplfigures()


#%% Panel A
cachedDataPath = cacheFolder / "decodeWithIncreasingNumberOfNeurons.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisDecoding.decodeWithIncreasingNumberOfNeurons(endoDataPath)
    decodingData.to_pickle(cachedDataPath)

decodingData.insert(1, "genotype", decodingData.session.str.split("_").str[0])

##%%
plt.sca(layout.axes["decodeWithIncreasingNumberOfNeurons"]["axis"])
for strSess, df in decodingData.groupby("session"):
    genotype = strSess.split("_")[0]
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=0.35, lw=.35)
    plt.plot(df.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
             alpha=0.35, lw=.35)
for genotype, df in decodingData.groupby("genotype"):
    gavg = df.groupby('nNeurons').realAccuracy.mean()
    gsem = df.groupby('nNeurons').realAccuracy.sem()
    plt.plot(gavg, color=style.getColor(genotype), alpha=1.0)
    plt.fill_between(gavg.index, gavg-gsem, gavg+gsem, lw=0,
                     color=style.getColor(genotype), alpha=.2, zorder=-99)
savg = decodingData.groupby('nNeurons').shuffledAccuracy.mean()
ssem = decodingData.groupby('nNeurons').shuffledAccuracy.sem()
plt.plot(savg, color=style.getColor("shuffled"), alpha=1.0)
plt.fill_between(savg.index, savg-ssem, savg+ssem, lw=0,
                 color=style.getColor('shuffled'), alpha=.2, zorder=-99)

order = ("d1", "a2a", "oprm1")
meanHandles = [mpl.lines.Line2D([], [], color=style.getColor(g), 
                                label={'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[g])
                   for g in order]
shuffleHandle = mpl.lines.Line2D([], [], color=style.getColor("shuffled"),
                                 label='shuffled')
plt.legend(handles=meanHandles+[shuffleHandle],
           bbox_to_anchor=(1,.28), loc='center right', ncol=2)

plt.ylim(0,1)
plt.xlim(0,200)
plt.xlabel("number of neurons")
plt.ylabel("decoding accuracy (%)")
plt.yticks((0,.5,1), (0,50,100))
plt.gca().set_yticks(np.arange(.25,1,.25), minor=True)
plt.xticks(np.arange(0,201,50))
plt.gca().set_xticks(np.arange(25,200,25), minor=True)
sns.despine(ax=plt.gca())


#%% Panel B
cachedDataPath = cacheFolder / "decodeConfusion.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisDecoding.decodingConfusion(endoDataPath)
    decodingData.to_pickle(cachedDataPath)
    
#means = confusionDiagonal.groupby("sess").mean()
#nNeurons = means.nNeurons
#labels = list(means.columns)
#for i in range(6):
#     labels[i] = labels[i][:4]
#genotypes = means.index.str.split("_").str[0]
    
decodingData["genotype"] = decodingData.sess.str.split("_").str[0]

for gt, data in decodingData.groupby("genotype"):
    #gtMeans = np.average(means[genotypes==gt].drop("nNeurons", axis=1), axis=0, weights=nNeurons[genotypes==gt])
    weightedData = data.set_index(["true", "predicted"]).eval("occurences * nNeurons")
    weightedData = weightedData.groupby(level=[0,1]).sum().unstack()
    weightedData /= weightedData.sum(axis=1)[:, np.newaxis]
    gtMeans = np.diag(weightedData)
    
    #cmap = {"oprm1": plt.cm.Greens, "d1": plt.cm.Reds, "a2a": plt.cm.Blues}[gt]
    #cmap = sns.light_palette(style.getColor(gt), 1024, as_cmap=True)
    cmap = cmocean.cm.algae_r
    labels = [(l[:4] if l[0]=='m' or l[1]=='C' else l) for l in weightedData.columns]
    di = {k: cmap(v) for k, v in zip(labels, gtMeans)}
    plt.sca(layout.axes["decodingAccuracyPerLabel_{}".format(gt)]["axis"])
    fancyViz.drawBinnedSchematicPlot(di, lw=mpl.rcParams['axes.linewidth'])

cax = layout.axes["decodingAccuracyCbar"]
cb1 = mpl.colorbar.ColorbarBase(cmap=cmap, ax=cax, norm=mpl.colors.Normalize(vmin=0, vmax=100),
                                orientation='horizontal', ticks=(0,50,100))
cb1.outline.set_visible(False)
#cax.tick_params(axis='both', which='both',length=0)
#cax.set_xlabel('decoding accuracy (%)', fontdict={'fontsize':6})
#cax.xaxis.set_label_position('top')
cax.set_axis_off()
cax.text(-.025, .25, 0, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, 100, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.125, 'decoding accuracy (%)', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)


#%% Panel C
sel_neurons = [92,44,16]
sel_colors = list(cmocean.cm.phase(i) for i in [.25,.5,.75])
examples = [('oprm1','5703',('190114','190116','190126'), neuron)
                for neuron in sel_neurons]

d1_rois = pd.read_hdf(endoDataPath, key='/rois/{}/{}/{}'.format(examples[0][0],
                                                                examples[0][1],
                                                                examples[0][2][0]))
d1_rois = np.array([d1_rois[n].unstack('x').values for n in d1_rois])

alignments= h5py.File(alignmentDataPath, 'r')
alignment_path = 'data/{}/{}/{}/{}'.format(examples[0][0],
                                           examples[0][1],
                                           examples[0][2][0],
                                           examples[0][2][2])
d2_rois = alignments[alignment_path + '/A_t'].value
match = alignments[alignment_path + '/match'].value
alignments.close()

colors = np.array([list(cmocean.cm.phase(i))[:-1] for i in np.random.rand(len(match))])
colors[np.isin(match[:,0], sel_neurons)] = np.array(sel_colors)[:,:-1]
for d, drois in enumerate((d1_rois, d2_rois)):
    rois_bg = drois 
    rois = drois[match[:,d]]
    sel_cnts = analysisDecoding.get_centers(rois_bg[{0:sel_neurons,
                                                     1:[dict(match)[n] for n in sel_neurons]}[d]])
    
    rs = []
    for roi, color in zip(rois, colors):
        roi /= roi.max()
        roi = roi**1.5
        roi = np.clip(roi-.1, 0, .85)
        roi /= roi.max()
        r = np.array([(roi > 0).astype('int')]*3) * color[:, np.newaxis, np.newaxis]
        r = np.concatenate([r, roi[np.newaxis]], axis=0)
        rs.append(r.transpose((1,2,0)))    
    rs = np.array(rs)
    
    rs_bg = []
    for roi in rois_bg:
        roi /= roi.max()
        roi = roi**1.5
        roi = np.clip(roi-.1, 0, .85)
        roi /= roi.max()
        rs_bg.append(roi)
    img_bg = np.array(rs_bg).max(axis=0)

    ax = layout.axes['alignment_example_d{}'.format(d+1)]['axis']    
    ax.imshow(img_bg, cmap='bone_r', alpha=.5, aspect='auto')
    for img in rs:
        ax.imshow(img, aspect='auto')
        
    ax.scatter(sel_cnts[:,0], sel_cnts[:,1], marker='o', edgecolor='k', facecolor='none', 
               s=25, alpha=1, lw=mpl.rcParams['axes.linewidth'])
    
    #ax.set_xlim((20,320))
    #ax.set_ylim((230,0))
    ax.axis('off')


##%% Panel E
alignmentStore = h5py.File(alignmentDataPath, "r")
def findAlignedNeuron(genotype, animal, fromDate, toDate, neuron):
    if fromDate == toDate:
        return neuron
    else:
        matches = alignmentStore["/data/{}/{}/{}/{}/match".format(genotype, animal, fromDate, toDate)]
        return pd.Series(matches[:,1], matches[:,0]).loc[neuron]

saturation = 1
for i in range(3):
    for j in range(3):
        sess = next(readSessions.findSessions(endoDataPath, animal=examples[i][1],
                                             date=examples[i][2][j], task="2choice"))
        neuron = findAlignedNeuron(examples[i][0], examples[i][1], examples[i][2][0],
                                   examples[i][2][j], examples[i][3])
        signal = sess.readDeconvolvedTraces()[neuron]
        signal -= signal.mean()
        signal /= signal.std()
        ax = layout.axes["acrossDays_ex{}{}".format(i+1,j+1)]["axis"]
        fv = fancyViz.SchematicIntensityPlot(sess, splitReturns=False,
                                             linewidth=mpl.rcParams['axes.linewidth'],
                                             saturation=saturation, smoothing=7)
        img = fv.draw(signal, ax=ax)
    
    axbg = layout.axes['acrossDays_ex{}1_bg'.format(i+1)]['axis']
    axbg.axvspan(-.055, -.03, .1, .93, color=sel_colors[i], alpha=1,
                 clip_on=False)
    axbg.set_xlim((0,1))
    axbg.set_axis_off()

cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-.025, .25, -saturation, ha='right', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(1.025, .25, saturation, ha='left', va='center', fontdict={'fontsize':6},
         transform=cax.transAxes)
cax.text(.5, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6},
         transform=cax.transAxes)


#%% Panel D
cachedDataPath = cacheFolder / "decodingAcrossDays.pkl"
if cachedDataPath.is_file():
    decodingAcrossDays = pd.read_pickle(cachedDataPath)
else:
    decodingAcrossDays = analysisDecoding.decodingAcrossDays(endoDataPath, alignmentDataPath)
    decodingAcrossDays.to_pickle(cachedDataPath)

def bootstrapSEM(values, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(values), len(values), replace=True)
        avgs.append(np.average(values.iloc[idx], weights=weights.iloc[idx]))
    return np.std(avgs)
    
#acrossDays = acrossDays.rename(columns={"sameDayShuffled": "nextDayScore", "nextDayScore": "sameDayShuffled"})
fromDate = pd.to_datetime(decodingAcrossDays.fromDate, format="%y%m%d")
toDate = pd.to_datetime(decodingAcrossDays.toDate, format="%y%m%d")
td = (toDate - fromDate).dt.days
decodingAcrossDays["dayDifference"] = td

selection = decodingAcrossDays.query("fromTask=='2choice' & toTask=='2choice'")
for i,l,h in ((0,1,3), (1,4,13), (2,14,100)):
    g = (selection.query("dayDifference >= {} & dayDifference <= {}".format(l,h))
                  .groupby(["animal", "fromDate", "toDate"]))

    perAnimal = g.mean()[['nNeurons', 'sameDayScore', 'nextDayScore',
                          'sameDayShuffled', 'nextDayShuffled']]
    perAnimal["genotype"] = g.genotype.first()


    scaledScore = perAnimal[['sameDayScore', 'nextDayScore',
                             'sameDayShuffled', 'nextDayShuffled']] * \
                  np.stack([perAnimal.nNeurons]*4, axis=1)
    
    perGenotype = scaledScore.groupby(perAnimal.genotype).sum()
    totalNeurons = perAnimal.groupby('genotype').nNeurons.sum()
    perGenotype /= np.stack([totalNeurons]*4, axis=1)

    for genotype in ('d1','a2a','oprm1'):
        plt.sca(layout.axes["{}_decodingAcrossDays_{}".format(genotype, i+1)]["axis"])
    
        for r in perAnimal.query("genotype == @genotype").itertuples():
            plt.plot([0,1], [r.nextDayScore, r.nextDayShuffled],
                     lw=r.nNeurons/150,#lw=mpl.rcParams['axes.linewidth'],
                     alpha=.25, clip_on=False, zorder=-99, color=style.getColor(genotype))
        
        r = perGenotype.loc[genotype]
        animalsWithGt = perAnimal.query("genotype == '{}'".format(genotype))
        nextDaySEM = bootstrapSEM(animalsWithGt.nextDayScore, animalsWithGt.nNeurons)
        nextDayShuffledSEM = bootstrapSEM(animalsWithGt.nextDayShuffled, animalsWithGt.nNeurons)
        plt.errorbar(0, r.nextDayScore, nextDaySEM,
                     c=style.getColor(genotype), clip_on=False,
                     marker='v', markersize=3.6, markerfacecolor='w',
                     markeredgewidth=.8)
        plt.errorbar(1, r.nextDayShuffled, nextDayShuffledSEM,
                     c=style.getColor(genotype), clip_on=False,
                     marker='o', markersize=3.2, markerfacecolor='w',
                     markeredgewidth=.8)
        plt.plot([0,1], [r.nextDayScore, r.nextDayShuffled],
                 color=style.getColor(genotype), clip_on=False)
        
        plt.ylim((0,1))
        plt.xlim((-.35,1))
        plt.xticks((.5,),('{}'.format({0:'1-3',1:'4-13',2:'14+'}[i]),))
        plt.yticks(())
        if i != 0:
            sns.despine(ax=plt.gca(), left=True, trim=False)
        else:
            plt.yticks((0,.5,1), ())
            plt.gca().set_yticks((.25,.75), minor=True)
            if genotype == 'd1':
                plt.yticks((0,.5,1),(0,50,100))
                plt.ylabel('decoder accuracy (%)')
        
            sns.despine(ax=plt.gca(), trim=False)
        if i == 1:
            plt.title({'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}[genotype])
            if genotype == 'a2a':
                plt.xlabel('days later')

axt = layout.axes['across_days_legend1']['axis']
legend_elements = [mpl.lines.Line2D([0], [0], marker='v', color='k', label='neural activity',
                                    markerfacecolor='w', markersize=3.6,
                                    markeredgewidth=.8),
                   mpl.lines.Line2D([0], [0], marker='o', color='k',
                                    label='neural activity\n(labels shuffled)',
                                    markerfacecolor='w', markersize=3.2,
                                    markeredgewidth=.8)
                  ]
axt.legend(handles=legend_elements, loc='center', ncol=1, mode='expand')
axt.axis('off')


#for i,l,h in ((0,1,3), (1,4,13), (2,14,100)):#(1,4,6), (2,7,14), (3,14,100)):
#    g = (selection.query("dayDifference >= {} & dayDifference <= {}".format(l,h))
#                  .groupby(["animal", "fromDate", "toDate"]))
#    
#    perAnimal = g.mean()[['nNeurons', 'sameDayScore', 'nextDayScore',
#                          'sameDayShuffled', 'nextDayShuffled']]
#    perAnimal["genotype"] = g.genotype.first()
#    
#    
#    scaledScore = perAnimal[['sameDayScore', 'nextDayScore']] * perAnimal.nNeurons[:,np.newaxis]
#    perGenotype = scaledScore.groupby(perAnimal.genotype).sum()
#    perGenotype /= perAnimal.groupby("genotype").nNeurons.sum()[:, np.newaxis]
#    
#    shuffleScore = perAnimal[['sameDayShuffled', 'nextDayShuffled']] * perAnimal.nNeurons[:,np.newaxis]
#    shuffleScore = shuffleScore.sum(axis=0) / perAnimal.nNeurons.sum()
#    
#    plt.sca(layout.axes["decodingAcrossDays_{}".format(i+1)]["axis"])
#    
#    for r in perAnimal.itertuples():
#        plt.plot([0,1], [r.sameDayScore, r.nextDayScore], lw=style.lw()*r.nNeurons/400.0,
#                 c=style.getColor(r.genotype), alpha=0.2)
#    for r in perGenotype.itertuples():
#        gt = r.Index
#        animalsWithGt = perAnimal.query("genotype == '{}'".format(gt))
#        sameDaySEM = bootstrapSEM(animalsWithGt.sameDayScore, animalsWithGt.nNeurons)
#        nextDaySEM = bootstrapSEM(animalsWithGt.nextDayScore, animalsWithGt.nNeurons)
#        plt.errorbar([0,1], [r.sameDayScore, r.nextDayScore], [sameDaySEM, nextDaySEM],
#                     lw=style.lw(), c=style.getColor(gt))
#        
#    plt.plot([0,1], [shuffleScore.sameDayShuffled, shuffleScore.nextDayShuffled],
#             lw=style.lw(), c=style.getColor("shuffled"))
#    
#    plt.ylim(0,1)
#    plt.xlim(-0.25, 1.25)
#    xlab = ("1-3 days\nlater", "4-13 days\nlater", "14+ days\nlater")
#    plt.xticks((0,1), ("Same\nday", xlab[i]))
#    if i==0:
#        plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
#        plt.ylabel("Decoding accuracy (%)")
#    else:
#        plt.yticks(np.linspace(0,1,5), [""]*5)
#    sns.despine(ax=plt.gca())


#%% Panel F
sess = next(readSessions.findSessions(endoDataPath, animal="5308", date="190131"))
lfa = sess.labelFrameActions(reward="sidePorts")
deconv = sess.readDeconvolvedTraces(zScore=True).reset_index(drop=True)
X = deconv[lfa.label=="mR2C-"]
Y = lfa.actionProgress[lfa.label=="mR2C-"]
avgActivity = X.groupby((Y*10).astype("int")/10.0).mean().T
sorting = avgActivity.idxmax(axis=1).argsort()
plt.sca(layout.axes["movementProgressRaster"]["axis"])
plt.imshow(avgActivity.iloc[sorting], aspect="auto",
           interpolation="nearest", vmin=-1, vmax=1, cmap="RdYlBu_r")
plt.xticks((-0.5,4.5,9.5), ("Right\nport", "Half-way", "Center\nport"))#, rotation=30, ha="right", va="top")#(0, 50, 100))
plt.yticks([0, len(sorting)-1], [len(sorting), 0])
plt.xlabel("Progress (%)")
plt.ylabel("Neuron (by peak)")

exampleNeurons = (7, 66, 13)
fv = fancyViz.SchematicIntensityPlot(sess, linewidth=style.lw()*0.5,
                                     splitReturns=False, smoothing=7)
for i in range(3):
    ax = layout.axes["movementExample{}".format(i+1)]["axis"]
    fv.draw(deconv[exampleNeurons[i]], ax=ax)

    
#%% Panel G
cachedDataPath = cacheFolder / "decodeMovementProgress_mR2C.pkl"
if cachedDataPath.is_file():
    decodingMovementProgress = pd.read_pickle(cachedDataPath)
else:
    decodingMovementProgress = analysisDecoding.decodeMovementProgress(endoDataPath)
    decodingMovementProgress.to_pickle(cachedDataPath)
    
def calcCorr(df):
    r = scipy.stats.pearsonr(df.true, df.predicted)[0]
    return pd.Series((r, df.nNeurons.iloc[0]), ("correlation", "nNeurons"))

exampleSession = decodingMovementProgress.query("sess == 'oprm1_5308_190131' & not shuffle")
means = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.mean()
stds = exampleSession.groupby(np.floor(exampleSession.true * 10)/10).predicted.std()
plt.sca(layout.axes["decodingProgressExample"]["axis"])
plt.plot([0,100], [0, 100], 'k--', alpha=0.2)
plt.errorbar(means.index*100, means*100, yerr=stds*100, fmt='.-', ms=10, color=style.getColor("oprm1"))
plt.xlim(-5,100)
plt.ylim(-5,100)
plt.xticks((0,50,100), ("Right\nport", "Half-way", "Center\nport"))#, rotation=30, ha="right", va="top")
plt.yticks((0,50,100), ("Right\nport", "Half-way", "Center\nport"))
plt.xlabel("Truth")
plt.ylabel("Decoded")
corr = calcCorr(exampleSession).loc["correlation"]
plt.text(100, 10, "r = {:.3f}".format(corr), fontsize=mpl.rcParams['font.size'], color="k", ha="right")
sns.despine(ax=plt.gca())


#%% Panel H
avgCorr = decodingMovementProgress.query("not shuffle").groupby("sess").apply(calcCorr)
avgCorr["genotype"] = avgCorr.index.str.split("_").str[0]
avgCorr["animal"] = avgCorr.index.str.split("_").str[1]
avgCorr["date"] = avgCorr.index.str.split("_").str[2]
avgCorr.sort_values(["genotype", "animal", "date"], ascending=False, inplace=True)
ax = layout.axes["movementProgressCorrelations"]["axis"]
sessionBarPlot.sessionBarPlot(avgCorr, yCol="correlation", weightCol="nNeurons",
                              ax=ax, colorFunc=style.getColor, weightScale=0.05)
#shuffledCorr = calcCorr(decodingMovementProgress.query("shuffle").set_index("sess")).to_frame()
#shuffledCorr["genotype"] = "shuffled"
#shuffledCorr["animal"] = shuffledCorr.index.str.split("_").str[1]
#shuffledCorr["date"] = shuffledCorr.index.str.split("_").str[2]
ax.set_ylim(0,1)
sns.despine(ax=ax)
ax.set_ylabel("Correlation\ntruth and decoded")


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "decoding.svg")

