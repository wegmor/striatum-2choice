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
import itertools
import subprocess
from matplotlib.ticker import MultipleLocator

import analysisOpenField, analysisTunings
import style
from utils import readSessions, fancyViz

style.set_context()
plt.ioff()

#%%

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "openFieldSupp.svg")
layout.make_mplfigures()
svgName = "openFieldSupp.svg"


genotypeNames = {'d1':'D1','a2a':'A2A','oprm1':'Oprm1'}
behaviorNames = {'stationary': 'stationary', 'running': 'running', 'leftTurn': 'left turn',
                 'rightTurn': 'right turn'}

#All 4-behavior panels    
segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)


## New panel A
tuningData = analysisOpenField.getTuningData(endoDataPath)
tuningData['signp'] = tuningData['pct'] > .995
tuningData['signn'] = tuningData['pct'] < .005
tuningData_shuffled = analysisOpenField.getTuningData_shuffled(endoDataPath)
tuningData_shuffled['signp'] = tuningData_shuffled['pct'] > .995
tuningData_shuffled['signn'] = tuningData_shuffled['pct'] < .005
for gt, behavior in itertools.product(genotypeNames.keys(), behaviorNames.keys()):
    ax = layout.axes['tuning_hist_{}_{}'.format(gt, behavior)]['axis']
    hdata = tuningData.query('genotype == "{}" & action == "{}"'.format(gt, behavior)).copy()
    shuffle_kde = tuningData_shuffled.query('genotype == "{}" & action == "{}"'.format(gt, behavior)).copy()
    sns.kdeplot(shuffle_kde['tuning'], ax=ax, color=style.getColor('shuffled'), alpha=.75,
                clip_on=False, zorder=10, label='')
    sns.kdeplot(hdata['tuning'], ax=ax, color='gray', alpha=.75, clip_on=True,
                zorder=-99, label='')
    bins = np.arange(-20.5, 20.5)
    none_hist = np.histogram(hdata.loc[~hdata['signp'], 'tuning'], bins=bins)[0] / len(hdata.tuning)
    sign_hist = np.histogram(hdata.loc[hdata['signp'], 'tuning'], bins=bins)[0] / len(hdata.tuning)
    ax.bar((bins+.5)[:-1], none_hist, lw=0, color='gray', alpha=.6)
    ax.bar((bins+.5)[:-1], sign_hist, lw=0, color=style.getColor(behavior), bottom=none_hist)
    #ax.text(20,0.05,'significant\ntuning',ha='right',va='bottom',fontdict={'fontsize':7},
    #        color=style.getColor(behavior))
    ax.text(0,0.435,behaviorNames[behavior]+' tuning',ha='center',va='center',
            fontdict={'fontsize':7})
    #ax.text(3,.25,'shuffled',ha='left',va='center',
    #        fontdict={'fontsize':7,'color':style.getColor('shuffled'),'alpha':1.0})
    ax.set_yticks((0,0.2,0.4))
    if behavior in ("leftTurn", "running"):
        ax.set_ylabel('density')
    else:
        ax.set_yticklabels(("","",""))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xticks(np.arange(-20,21,10))
    if behavior in ("running", "stationary"):
        ax.set_xlabel('tuning score')
    else:
        ax.set_xticklabels([""]*5)
    ax.set_xlim((-20,20))
    ax.set_ylim((0,0.4))
    sns.despine(ax=ax)



## Panel A
tuningData = analysisOpenField.getTuningData(endoDataPath, segmentedBehavior)
df = tuningData.copy()

df['signp'] = df['pct'] > .995
df['signn'] = df['pct'] < .005
df['sign'] = df.signp.astype('int') - df.signn.astype('int')

sign_count = (df.groupby(['genotype','animal','date','action'])
                .agg({'signp':'sum','signn':'sum'}))
total_count = (df.groupby(['genotype','animal','date','action'])
                 [['signp','signn']].count())
sign_pct = sign_count / total_count
sign_pct['noNeurons'] = total_count.signp

order = ["stationary", "running", "leftTurn", "rightTurn"]

# v x coords for actions
a2x = dict(zip(order, np.arange(.5,12)))
sign_pct['x'] = sign_pct.reset_index().action.replace(a2x).values
# v color for actions
sign_pct['color'] = sign_pct.reset_index().action.apply(style.getColor).values

for tuning in ('signp','signn'):
    for g, gdata in sign_pct.groupby('genotype'):
        ax = layout.axes['fracTuned_{}_{}'.format(g,{'signp':'pos','signn':'neg'}[tuning])]['axis']
        ax.scatter(analysisOpenField.jitter(gdata.x, .15), gdata[tuning], s=gdata.noNeurons/20,
                   edgecolor=gdata.color, facecolor='none', clip_on=False)
        
        avg = gdata.groupby('x').apply(analysisOpenField.wAvg, tuning, 'noNeurons')
        sem = gdata.groupby('x').apply(analysisOpenField.bootstrap, tuning, 'noNeurons')
        ax.errorbar(avg.index, avg, sem, fmt='.-', c='k')
        
        ax.set_xticks(np.arange(.5,4))
        ax.set_xlim((0,4))
        ax.set_xticklabels([behaviorNames[b] for b in order], rotation=45, ha="right")
        title = genotypeNames[g]
        title += "\n" + {'signp':'positively','signn':'negatively'}[tuning]
        ax.set_title(title)
        ax.set_ylabel('')
        ax.set_yticks((0,.5,1))
        ax.set_yticklabels(())
        if g == 'd1' and tuning == 'signp':
            ax.set_ylabel('tuned neurons (%)')
            ax.set_yticklabels((0,50,100))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_ylim((0,1))
        
        sns.despine(ax=ax)
        
## Panel B
tunings = tuningData.set_index(["genotype", "animal", "date", "neuron", "action"]).tuning

cax = layout.axes['corr_colorbar']['axis']
cax.tick_params(axis='x', which='both',length=0)

for genotype in ("oprm1", "d1", "a2a"):
    corr = tunings.loc[genotype].unstack()[order].corr()
    ax = layout.axes["corrMatrix_{}".format(genotype)]["axis"]
    yticks = [behaviorNames[b] for b in order] if genotype == "d1" else False
    hm = sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, annot=True, fmt=".2f",
                     cmap=cmocean.cm.balance, cbar=True, cbar_ax=cax,
                     cbar_kws={'ticks':(-1,0,1), 'orientation': 'horizontal'},
                     annot_kws={'fontsize': 4.0}, yticklabels=yticks,
                     xticklabels=[behaviorNames[b] for b in order],
                     linewidths=mpl.rcParams["axes.linewidth"])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(genotypeNames[genotype], pad=3)
    ax.set_ylim(4, 0)
    ax.tick_params("both", length=0, pad=3)

##
class OpenFieldLegend(fancyViz.OpenFieldSchematicPlot): 
    def fakeBehavior(self):
        fake  = [(i, i*100, 100, i+20, "leftTurn") for i in range(150)]
        fake += [(i+150, (i+150)*100, 100, -i-20, "rightTurn") for i in range(150)]
        fake += [(i+150*2, 150*2*100 + i*20, 20, 0, "running") for i in range(40)]
        sb =  pd.DataFrame(fake, columns=["actionNo", "startFrame", "numFrames",
                                           "netTurn", "behavior"])
        lastFrame = sb.startFrame.iloc[-1] + sb.numFrames.iloc[-1]
        #sb = sb[["actionNo", "startFrame", "numFrames", "netTurn", "behavior"]]
        sb["nextBehavior"] = sb.behavior.shift(-1)
        sb = sb.set_index("startFrame").reindex(np.arange(lastFrame), method="ffill")
        sb["progress"] = sb.groupby("actionNo").cumcount() / sb.numFrames
        m = sb.behavior=="leftTurn"
        rad = 0.5 + 0.5*(sb[m].netTurn / 150.0)
        ang = np.deg2rad(sb[m].progress * sb[m].netTurn)
        sb.loc[m, "x"] = -1.5 + rad*np.cos(ang)
        sb.loc[m, "y"] = rad*np.sin(ang)
        m = sb.behavior=="rightTurn"
        rad = 0.5 + 0.5*(-sb[m].netTurn / 150.0)
        ang = np.deg2rad(-180 + sb[m].progress * sb[m].netTurn)
        sb.loc[m, "x"] = 1.5 + rad*np.cos(ang)
        sb.loc[m, "y"] = rad*np.sin(ang)
        m = sb.behavior=="running"
        sb.loc[m, "x"] = 0
        sb.loc[m, "y"] = 2*sb[m].progress
        m = sb.behavior=="stationary"
        sb.loc[m, "y"] = 0.75*(sb[m].progress-1)
        m = np.logical_and(sb.behavior=="stationary", sb.nextBehavior=="running")
        sb.loc[m, "x"] = 0
        m = np.logical_and(sb.behavior=="stationary", sb.nextBehavior=="leftTurn")
        sb.loc[m, "x"] = -0.5 * sb[m].progress
        m = np.logical_and(sb.behavior=="stationary", sb.nextBehavior=="rightTurn")
        sb.loc[m, "x"] = 0.5 * sb[m].progress
        self.rawCoords = sb
        
        schematicCoord = sb[["x", "y"]]*50
        schematicCoord["x"] += 150
        schematicCoord["y"] += 50
        self.coordinates = schematicCoord.values
        return sb
ofl = OpenFieldLegend(cmap="Greens",
                      linewidth=mpl.rcParams['axes.linewidth'])
fakeBehavior = ofl.fakeBehavior()
absTurn = fakeBehavior.netTurn.abs()
progress = fakeBehavior.progress
turn = absTurn*progress
ofl.draw(turn/75.0-1, layout.axes['legend_ex1']['axis'])
ofl.draw(absTurn/75.0-1, layout.axes['legend_ex2']['axis'])
ofl.draw(progress*2-1, layout.axes['legend_ex3']['axis'])
for i in range(3):
    cax = layout.axes["legend_ex{}_colorbar".format(i+1)]
    if i<=1: ticks = (0, 75, 150)
    else: ticks = (0, 50, 100)
    cb1 = mpl.colorbar.ColorbarBase(cmap=mpl.cm.Greens, ax=cax,
                                    norm=mpl.colors.Normalize(vmin=ticks[0], vmax=ticks[-1]),
                                    orientation='vertical', ticks=ticks)
    cb1.outline.set_visible(False)
    cax.set_axis_off()
    for tick in ticks:
        cax.text(-0.5, tick/ticks[-1], tick, ha='right', va='center', fontsize=6,
                 transform=cax.transAxes)
    label = ("turned\nangle ($\circ\:$)", "total angle\nof turn ($\circ\:$)",
             "turn\nprogress (%)")[i]
    cax.text(3.5, 0.5, label, ha='center', va='center', fontdict={'fontsize':6},
             transform=cax.transAxes, rotation=90)

examples = {
    'd1':   [('d1_5652_190203', 1),  ('d1_5652_190202', 16),
             ('d1_5643_190224', 66), ('d1_5643_190201', 17),
             ('d1_5643_190201', 20)],
    'a2a':  [('a2a_6043_190224', 192), ('a2a_6043_190130', 58),
             ('a2a_5693_190224', 13), ('a2a_5693_190224', 38),
             ('a2a_6043_190224', 47)],
    'oprm1':[('oprm1_5703_190224', 2), ('oprm1_5703_190201', 39),
             ('oprm1_5308_190204', 69), ('oprm1_5308_190224', 71),
             ('oprm1_5703_190130', 54)]
}
for gt in ("d1", "a2a", "oprm1"):
    for i in range(5):
        genotype, animal, date = examples[gt][i][0].split("_")
        neuron = examples[gt][i][1]
        sess = next(readSessions.findSessions(endoDataPath, animal=animal,
                                                       date=date, task="openField"))
        deconv = sess.readDeconvolvedTraces()[neuron]
        deconv -= deconv.mean()
        deconv /= deconv.std()
        ax = layout.axes["ex_of_{}_{}".format(gt, i+1)]["axis"]
        fvof = fancyViz.OpenFieldSchematicPlot(sess, linewidth=style.lw()*0.5)
        img = fvof.draw(deconv, ax=ax)
    
cax = layout.axes['colorbar_examples']['axis']
cb = plt.colorbar(img, cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(-1.05, -.3, '-1', ha='right', va='center', fontdict={'fontsize':6})
cax.text(1.05, -.3, '1', ha='left', va='center', fontdict={'fontsize':6})
cax.text(0, 1.1, 'z-score', ha='center', va='bottom', fontdict={'fontsize':6})
    

layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / svgName)
subprocess.check_call(['inkscape', '-f', outputFolder / svgName,
                                   '-A', outputFolder / (svgName[:-3]+'pdf')])