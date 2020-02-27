import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import figurefirst

from utils import fancyViz
from utils import readSessions
from utils import sessionBarPlot
import analysisDecoding
import style

style.set_context()

endoDataPath = pathlib.Path('data') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('data') / "alignment_190227.hdf"
outputFolder = pathlib.Path("svg")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "staySwitch.svg")
layout.make_mplfigures()


#Panel B
examples = [("5308", "190131", 292, "oprm1"),
            ("5703", "190114", 167, "oprm1"),
            ("5574", "190126", 13, "oprm1"),
            ("5574", "190126", 32, "oprm1"),
            ("5308", "190131", 27, "oprm1"),
            ("5308", "190131", 36, "oprm1"),
            
            ("5643", "190114", 178, "d1"),
            ("5652", "190128", 18, "d1"),
            ("5643", "190114", 16, "d1"),
            ("5643", "190114", 18, "d1"),
            ("5643", "190114", 96, "d1"),
            ("5651", "190203", 10, "d1"),
            
            ("5693", "190115", 98, "a2a"),
            ("5693", "190115", 170, "a2a"),
            ("5693", "190115", 258, "a2a"),
            ("5693", "190115", 284, "a2a"),
            ("5693", "190115", 340, "a2a"),
            ("5693", "190115", 405, "a2a")]
for i in range(18):
    sess = next(readSessions.findSessions(endoDataPath, animal=examples[i][0],
                                          date=examples[i][1], task="2choice"))
    neuron = examples[i][2]
    signal = sess.readDeconvolvedTraces(rScore=True)[neuron]
    ax = layout.axes["example_{}_{}".format(examples[i][3], (i%6)+1)]["axis"]
    lw = matplotlib.rcParams["lines.linewidth"]*0.5
    fv = fancyViz.SwitchSchematicPlot(sess, linewidth=lw)
    img = fv.draw(signal, ax=ax)
cax = layout.axes['colorbar']['axis']
cb = plt.colorbar(img, cax=cax, orientation='vertical')
cb.outline.set_visible(False)
cax.set_axis_off()
cax.text(.5, -.15, -1, ha='center', va='bottom', fontdict={'fontsize':6},
        transform=cax.transAxes)
cax.text(.5, 1.15, 1, ha='center', va='top', fontdict={'fontsize':6},
        transform=cax.transAxes)
cax.text(-1.05, 0.5, 'z-score', ha='center', va='center', fontdict={'fontsize':6},
        rotation=90, transform=cax.transAxes)


layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "staySwitch.svg")