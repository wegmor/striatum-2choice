#import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import matplotlib
import pathlib
import figurefirst
import tqdm
#import subprocess

from utils import fancyViz
from utils import readSessions
import style

style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
    
svgName = "continuity.svg"
layout = figurefirst.FigureLayout(templateFolder / "continuity.svg")
layout.make_mplfigures()
selection = pd.read_csv("continuitySelection.csv", comment="#")
fancyVizs = {}
signals = {}
lw = matplotlib.rcParams["axes.linewidth"]
uniqueSessions = selection.session.unique()
with tqdm.tqdm(total=len(uniqueSessions), desc="Loading data") as t:
    for sess in readSessions.findSessions(endoDataPath, task="2choice"):
        if str(sess) in uniqueSessions:
            k = str(sess)
            neurons = selection[selection.session == str(sess)].neuron
            fancyVizs[k] = fancyViz.SchematicIntensityPlot(sess, smoothing=7, linewidth=lw, splitReturns=False)
            signals[k] = sess.readDeconvolvedTraces(rScore=True)[neurons]
            t.update(1)

for i, (session, neuron) in selection.iterrows():
    if i>=14: break
    ax = layout.axes["s{}".format(i+1)]["axis"]
    fancyVizs[session].draw(signals[session][neuron], ax=ax)
    genotype, animal, date = session.split("_")
    ax.set_title("#{} ({})\nneuron {}".format(animal, date, neuron), fontsize=7)
    #if (i//6)%2 == 0:
    #    ax.plot([0,0], [-2.75, -2.25], color=style.getColor(genotype), lw=2)
    #else:
    #    ax.plot([0,0], [2.5, 2.0], color=style.getColor(genotype), lw=2)
    #axs.flat[i].set_title("{}, #{}".format(session, neuron))

layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / svgName)

#subprocess.check_call(['inkscape', '-f', outputFolder / svgName,
#                                   '-A', outputFolder / (svgName[:-3]+'pdf')])