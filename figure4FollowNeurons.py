import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import pathlib
import figurefirst
import cmocean
from utils import readSessions, fancyViz, alluvialPlot
import analysisOpenField
import analysisTunings
import analysisDecoding
import analysisOftVs2Choice
import style
import subprocess
style.set_context()
plt.ioff()

#%%
endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path('svg')
templateFolder = pathlib.Path('templates')

if not outputFolder.is_dir():
    outputFolder.mkdir()

#%%
svgName = 'figure4FollowNeurons.svg'
layout = figurefirst.FigureLayout(templateFolder / svgName)
layout.make_mplfigures()

#%%
genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}

def bootstrapSEM(values, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(values), len(values), replace=True)
        avgs.append(np.average(values.iloc[idx], weights=weights.iloc[idx]))
    return np.std(avgs)


#%% remapping example neurons plot
ofSess =  next(readSessions.findSessions(endoDataPath, task='openField',
                                         animal='5308', date='190201'))
chSess = next(readSessions.findSessions(endoDataPath, task='2choice',
                                        animal=ofSess.meta.animal,
                                        date=ofSess.meta.date))
ofTraces = ofSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
chTraces = chSess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
ofTracking = analysisOftVs2Choice.getSmoothedTracking(endoDataPath, ofSess.meta.genotype, 
                                                      ofSess.meta.animal, ofSess.meta.date,
                                                      'openField')
chTracking = analysisOftVs2Choice.getSmoothedTracking(endoDataPath, ofSess.meta.genotype,
                                                      ofSess.meta.animal, ofSess.meta.date,
                                                      '2choice')

for n,neuron in enumerate([86,192][::-1]):
    ofTrace = ofTraces[neuron]
    chTrace = chTraces[neuron]

    ### open field
    # schematic
    schemAx = layout.axes[('n{}'.format(n),'ofSchematic')]['axis']
    fig = schemAx.get_figure()
    fig.sca(schemAx)
    fv = fancyViz.OpenFieldSchematicPlot(ofSess, linewidth=mpl.rcParams['axes.linewidth'],
                                         smoothing=3, saturation=1)
    img = fv.draw(ofTrace, ax=schemAx)
    
    # top events
    topTenEventsAx = layout.axes[('n{}'.format(n),'ofTopTen')]['axis']
    topTenTracesAxOf = layout.axes[('n{}'.format(n),'ofTopTenTraces')]['axis']
    pIdx = analysisOftVs2Choice.plotTop10Events(ofTrace, ofTracking, 
                                                axs=[topTenEventsAx, topTenTracesAxOf],
                                                offset=10)
    topTenEventsAx.vlines(-7, -15, -10, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenEventsAx.hlines(-15, -7, -2, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxOf.vlines(-2.5, -5, 5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxOf.hlines(-5, -2.5, 2.5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    
    # map
    mapAx = layout.axes[('n{}'.format(n),'ofMap')]['axis']
    fig.sca(mapAx)
    fv = fancyViz.TrackingIntensityPlot(session=ofSess, smoothing=20, saturation=1,
                                        drawBg=False)
    fv.draw(ofTrace, ax=mapAx)
    headCoords = fv.coordinates
    mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1], marker='x',
                  linewidth=mpl.rcParams['axes.linewidth'],
                  c=cmocean.cm.phase(np.arange(10)/11))
    
    ### 2-choice
    # schematic
    schemAx = layout.axes[('n{}'.format(n),'chSchematic')]['axis']
    fig.sca(schemAx)
    fv = fancyViz.SchematicIntensityPlot(chSess, linewidth=mpl.rcParams['axes.linewidth'],
                                         smoothing=5, splitReturns=False, saturation=1)
    img = fv.draw(chTrace, ax=schemAx)
    
    # top events
    topTenEventsAx = layout.axes[('n{}'.format(n),'chTopTen')]['axis']
    topTenTracesAxCh = layout.axes[('n{}'.format(n),'chTopTenTraces')]['axis']
    pIdx = analysisOftVs2Choice.plotTop10Events(chTrace, chTracking, 
                                                axs=[topTenEventsAx, topTenTracesAxCh],
                                                offset=10)
    topTenEventsAx.vlines(-7, -15, -10, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenEventsAx.hlines(-15, -7, -2, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxCh.vlines(-2.5, -5, 5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    topTenTracesAxCh.hlines(-5, -2.5, 2.5, lw=mpl.rcParams['axes.linewidth'], clip_on=False)
    
    # map
    mapAx = layout.axes[('n{}'.format(n),'chMap')]['axis']
    fig.sca(mapAx)
    fv = fancyViz.TrackingIntensityPlot(session=chSess, smoothing=20, saturation=1,
                                        portsUp=True, drawBg=False)
    fv.draw(chTrace, ax=mapAx)
    headCoords = fv.coordinates
    mapAx.scatter(headCoords[pIdx,0], headCoords[pIdx,1],
                  marker='x', linewidth=mpl.rcParams['axes.linewidth'],
                  c=cmocean.cm.phase(np.arange(10)/11),
                  transform=fv.transform)
    
    traceAxLims = np.concatenate([topTenTracesAxCh.get_ylim(),topTenTracesAxOf.get_ylim()])
    traceAxLims = (traceAxLims.min(), traceAxLims.max())
    topTenTracesAxCh.set_ylim(traceAxLims)
    topTenTracesAxOf.set_ylim(traceAxLims)

cax = layout.axes['second_colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})


#%% oft vs choice left/right tuning comparison
ofTuningData = analysisOpenField.getTuningData(endoDataPath)
chTuningData = analysisTunings.getTuningData(endoDataPath)

tuningData = (pd.concat([chTuningData, ofTuningData], keys=['choice','oft'], names=['task'])
                .reset_index('task').reset_index(drop=True))
tuningData['signp'] = tuningData.pct > .995
tuningData = tuningData.set_index(['task','action','genotype','animal','date','neuron'])

signTuned = tuningData.unstack(['task','action']).dropna()['signp']
#signTuned[('choice','none')] = signTuned.choice.sum(axis=1) == 0
#signTuned[('choice','port')] = signTuned.choice.loc[:,signTuned.choice.columns.str.contains('^p|^d')].any(axis=1)
signTuned[('choice','leftTurn')] = signTuned.choice[['mC2L-','mR2C-']].any(axis=1)
signTuned[('choice','rightTurn')] = signTuned.choice[['mC2R-','mC2R-']].any(axis=1)
    
for tuning in ['leftTurn', 'rightTurn']:
    data = signTuned.loc[:,signTuned.columns.get_level_values(1) == tuning].droplevel(1, axis=1)
    
    noNeurons = data.groupby(['genotype','animal','date']).size()
    pctTund = data.groupby(['genotype','animal','date']).sum(axis=1) / noNeurons.values[:,np.newaxis]
    pctTund *= 100
    
    obsOvlp = ((data.oft * data.choice).groupby(['genotype','animal','date']).sum() / \
                noNeurons)
    expOvlp = ((data.oft.groupby(['genotype','animal','date']).sum() / \
                noNeurons) * \
               (data.choice.groupby(['genotype','animal','date']).sum() / \
                noNeurons))
    ovlpDf = pd.concat([obsOvlp, expOvlp], axis=1, keys=['observed','expected']) * 100
    
    axs = [layout.axes['{}PctTuned'.format(tuning)]['axis'], 
           layout.axes['{}Ovlp'.format(tuning)]['axis']]
    
    for gt,df in pctTund.groupby('genotype'):
        x = {'d1':0,'a2a':1,'oprm1':2}[gt]
        ax = axs[0]
        means = np.average(df[['oft','choice']], axis=0, weights=noNeurons.loc[gt])
        sems = [bootstrapSEM(df.oft, noNeurons.loc[gt]),
                bootstrapSEM(df.choice, noNeurons.loc[gt])]
        ax.errorbar(x=x-.2, y=means[0], yerr=sems[0],
                    marker='^', color=style.getColor(gt), clip_on=False)
        ax.errorbar(x=x+.2, y=means[1], yerr=sems[1],
                     marker='s', color=style.getColor(gt), clip_on=False)
        ax.plot([[x-.2]*len(df),[x+.2]*len(df)], df.T.values,
                color=style.getColor(gt), alpha=.25, zorder=-99, clip_on=False)
        ax.axhline(0, ls=':', alpha=.5, color='k', zorder=-100,
                   lw=mpl.rcParams['axes.linewidth'])
    
    for gt,df in ovlpDf.groupby('genotype'):
        x = {'d1':0,'a2a':1,'oprm1':2}[gt]
        ax = axs[1]
        means = np.average(df[['observed','expected']], axis=0, weights=noNeurons.loc[gt])
        sems = [bootstrapSEM(df.observed, noNeurons.loc[gt]),
                bootstrapSEM(df.expected, noNeurons.loc[gt])]
        ax.errorbar(x=x-.2, y=means[0], yerr=sems[0],
                    marker='o', color=style.getColor(gt), clip_on=False)
        ax.errorbar(x=x+.2, y=means[1], yerr=sems[1],
                    marker='o', markeredgecolor=style.getColor(gt),
                    markerfacecolor='w', ecolor=style.getColor(gt), clip_on=False)
        ax.plot([[x-.2]*len(df),[x+.2]*len(df)], df.T.values,
                color=style.getColor(gt), alpha=.25, zorder=-99, clip_on=False)
        ax.axhline(0, ls=':', alpha=.5, color='k', zorder=-100,
                   lw=mpl.rcParams['axes.linewidth'])

    for ax in axs:
        ax.set_ylim(0,100)
        ax.set_yticks([0,50,100])
        ax.set_yticks([25,75], minor=True)
        ax.set_xlim((-.5,2.5))
        ax.set_xticks([])
        sns.despine(ax=ax, bottom=True)
        if tuning == 'rightTurn': ax.set_yticklabels([])
    if tuning == 'leftTurn':
        legend_elements = [mpl.lines.Line2D([0], [0], marker='^', ls='', color='k',
                                            label='open field'),
                           mpl.lines.Line2D([0], [0], marker='s', ls='', color='k',
                                            label='2-choice')]
        axs[0].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(.5,.95))
        axs[0].set_ylabel('% tuned', labelpad=0)
        legend_elements = [mpl.lines.Line2D([0], [0], marker='o', ls='', color='k',
                                            label='observed'),
                           mpl.lines.Line2D([0], [0], marker='o', ls='', markeredgecolor='k',
                                            markerfacecolor='w', label='expected')]
        axs[1].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(.5,.95))
        axs[1].set_ylabel('% overlap', labelpad=0)
        axs[0].set_title('left turn')
    else:
        axs[0].set_title('right turn')

axt = layout.axes['gt_legend']['axis']
legend_elements = [mpl.patches.Patch(color=style.getColor(gt), alpha=1,
                                     label=genotypeNames[gt])
                   for gt in ['d1','a2a','oprm1']
                  ]
axt.legend(handles=legend_elements, ncol=3, loc='center')
axt.axis('off')


#%% alluvial plot
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
openFieldTunings = analysisOpenField.getTuningData(endoDataPath, segmentedBehavior)

twoChoiceTunings = analysisTunings.getTuningData(endoDataPath)

for t in (twoChoiceTunings, openFieldTunings):
    t['signp'] = t['pct'] > .995
    t['signn'] = t['pct'] < .005
    
primaryTwoChoice = twoChoiceTunings.loc[twoChoiceTunings.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
primaryTwoChoice.loc[~primaryTwoChoice.signp, 'action'] = 'none'
primaryOpenField = openFieldTunings.loc[openFieldTunings.groupby(['genotype','animal','date','neuron']).tuning.idxmax()]
primaryOpenField.loc[~primaryOpenField.signp, 'action'] = 'none'

primaryPairs = primaryOpenField.join(primaryTwoChoice.set_index(["animal", "date", "neuron"]),
                                     on=["animal", "date", "neuron"], rsuffix="_2choice", how="inner")
primaryPairs = primaryPairs[["genotype", "animal", "date", "neuron", "action", "action_2choice"]]
primaryPairs.rename(columns={'action': 'action_openField'}, inplace=True)

order_openField = ["stationary", "running", "leftTurn", "rightTurn", "none"]
order_twoChoice = ["mC2L-", "mC2R-", "mL2C-", "mR2C-", "dL2C-", "pL2Co", "pL2Cr",
                   "pC2L-", "pC2R-", "dR2C-", "pR2Co", "pR2Cr", "none"]
primaryPairs.action_2choice = pd.Categorical(primaryPairs.action_2choice, order_twoChoice)
primaryPairs.action_openField = pd.Categorical(primaryPairs.action_openField, order_openField)

colormap = {a: style.getColor(a[:4]) for a in primaryPairs.action_2choice.unique()}
colormap.update({a: style.getColor(a) for a in primaryPairs.action_openField.unique()})

genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn', 'none': 'untuned'}
phaseNames = {'mC2L': 'center-to-left', 'mC2R': 'center-to-right', 'mL2C': 'left-to-center',
              'mR2C': 'right-to-center', 'pC': 'center port', 'pL2C': 'left port',
              'pR2C': 'right port', 'none': 'untuned'}

for gt in ("d1", "a2a", "oprm1"):
    ax = layout.axes['alluvial_{}'.format(gt)]['axis']
    data = primaryPairs.query("genotype == '{}'".format(gt))
    leftY, rightY = alluvialPlot.alluvialPlot(data, "action_openField",
                                              "action_2choice", colormap, ax, 
                                              colorByRight=False, alpha=0.75)
    ax.set_xlim(-0.2,1.2)
    ax.axis("off")
    ax.set_title(genotypeNames[gt])
    if gt=="d1":
        ticks = leftY.groupby(level=0).agg({'lower': np.min, 'higher': np.max}).mean(axis=1)
        for label, y in ticks.items():
            ax.text(-0.25, y, behaviorNames[label], ha="right", va="center",
                    color=style.getColor(label))
    if gt=="oprm1":
        newLabels = rightY.index.get_level_values(1).str[:4]
        newLabels = newLabels.str.replace('pC2.', 'pC')
        newLabels = newLabels.str.replace('d', 'p')
        ticks = rightY.groupby(newLabels).agg({'lower': np.min, 'higher': np.max}).mean(axis=1)
        for label, y in ticks.items():
            ax.text(1.25, y, phaseNames[label], ha="left", va="center",
                    color=style.getColor(label))
print("alluvial (followed neurons):")
print(primaryPairs.groupby("genotype").size())


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
    base = ax.transData
    rot = mpl.transforms.Affine2D().rotate_deg(-45)
    ax.imshow(img_bg, cmap='bone_r', vmin=0, vmax=1, alpha=.5, aspect='auto', transform=rot+base)
    ax.axis("equal")
    ax.set_xlim((100, 325))
    ax.set_ylim((160, -170))
    for img in rs:
        ax.imshow(img, aspect='auto', transform=rot+base)
    ax.scatter(sel_cnts[:,0], sel_cnts[:,1], marker='o', edgecolor='k',
               facecolor='none', s=10, alpha=1, lw=mpl.rcParams['axes.linewidth'],
               transform=rot+base)
    ax.axis('off')


##%% Panel C
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


#%% Panel E
decodingAcrossDays = analysisDecoding.decodingAcrossDays(endoDataPath, alignmentDataPath)

fromDate = pd.to_datetime(decodingAcrossDays.fromDate, format="%y%m%d")
toDate = pd.to_datetime(decodingAcrossDays.toDate, format="%y%m%d")
td = (toDate - fromDate).dt.days
decodingAcrossDays["dayDifference"] = td
selection = decodingAcrossDays.query("fromTask=='2choice' & toTask=='2choice'")

for i,l,h in ((0,1,3), (1,4,13), (2,14,100)):
    g = selection.query("dayDifference >= {} & dayDifference <= {}".format(l,h)).groupby(["animal", "fromDate", "toDate"])
    
    perAnimal = g.mean()[['nNeurons', 'sameDayScore', 'nextDayScore', 'sameDayShuffled', 'nextDayShuffled']]
    perAnimal["genotype"] = g.genotype.first()
    
    scaledScore = perAnimal[['sameDayScore', 'nextDayScore']] * perAnimal.nNeurons[:,np.newaxis]
    perGenotype = scaledScore.groupby(perAnimal.genotype).sum()
    perGenotype /= perAnimal.groupby("genotype").nNeurons.sum()[:, np.newaxis]
    
    shuffleScore = perAnimal[['sameDayShuffled', 'nextDayShuffled']] * perAnimal.nNeurons[:,np.newaxis]
    shuffleScore = shuffleScore.sum(axis=0) / perAnimal.nNeurons.sum()
    
    plt.sca(layout.axes["decodingAcrossDays_{}".format(i+1)]["axis"])
    
    for r in perAnimal.itertuples():
        plt.plot([0,1], [r.sameDayScore, r.nextDayScore], lw=style.lw()*r.nNeurons/400.0,
                 c=style.getColor(r.genotype), alpha=0.2)
    for r in perGenotype.itertuples():
        gt = r.Index
        animalsWithGt = perAnimal.query("genotype == '{}'".format(gt))
        sameDaySEM = bootstrapSEM(animalsWithGt.sameDayScore, animalsWithGt.nNeurons)
        nextDaySEM = bootstrapSEM(animalsWithGt.nextDayScore, animalsWithGt.nNeurons)
        plt.errorbar([0,1], [r.sameDayScore, r.nextDayScore], [sameDaySEM, nextDaySEM],
                     lw=style.lw(), c=style.getColor(gt))
    
    sameDayShuffledSEM = bootstrapSEM(perAnimal.sameDayShuffled, perAnimal.nNeurons)
    nextDayShuffledSEM = bootstrapSEM(perAnimal.nextDayShuffled, perAnimal.nNeurons)
    plt.errorbar([0,1], [shuffleScore.sameDayShuffled, shuffleScore.nextDayShuffled],
                 [sameDayShuffledSEM, nextDayShuffledSEM],
                 lw=style.lw(), c=style.getColor("shuffled"))
    
    plt.ylim(0,1)
    plt.xlim(-0.25, 1.25)
    #xlab = ("1-3 days later", "4-13 days later", "14+ days later")
    #plt.xticks((0,1), ("same day", xlab[i]), rotation=90)
    plt.xticks([])
    plt.xlabel(("1-3", "4-13", "14+")[i] + "\ndays", labelpad=7, fontsize=6)
    if i==0:
        plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
        plt.ylabel("decoding accuracy (%)")
    else:
        plt.yticks([])#np.linspace(0,1,5), [""]*5)
    plt.axhline(0, color='k', lw=0.5, alpha=0.5, ls=":")
    sns.despine(ax=plt.gca(), left=(i!=0), bottom=True)
axt = layout.axes['decodingAcrossDays_2']['axis']
genotypeNames["shuffled"] = "shuffled"
legend_elements = [mpl.lines.Line2D([0], [0], color=style.getColor(g), label=genotypeNames[g]) for g in ("d1", "a2a", "oprm1", "shuffled")]
axt.legend(handles=legend_elements, loc=(-0.7, 1.08), ncol=2)

print("Panel E:")
print(selection.groupby([pd.cut(selection.dayDifference, (1, 4, 14, 100)), "genotype"]).size())
#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / svgName)
subprocess.check_call(['inkscape', '-f', outputFolder / svgName,
                                   '-A', outputFolder / (svgName[:-3]+'pdf')])

