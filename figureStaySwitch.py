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

endoDataPath = "endoData_2019.hdf"

outputFolder = pathlib.Path("svg")
cacheFolder = pathlib.Path("cache")
templateFolder = pathlib.Path(__file__).parent / "templates"

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

layout = figurefirst.FigureLayout(templateFolder / "staySwitch.svg")
layout.make_mplfigures()


#Panel B
examples = [("5308", "190131", 292, "oprm1"),
            ("5703", "190114", 167, "oprm1"),
            ("5643", "190114", 178, "d1"),
            ("5652", "190128", 18, "d1"),
            ("5693", "190115", 340, "a2a"),
            ("5693", "190115", 284, "a2a")]
for i in range(6):
    sess = next(readSessions.findSessions(endoDataPath, animal=examples[i][0],
                                          date=examples[i][1], task="2choice"))
    neuron = examples[i][2]
    signal = sess.readDeconvolvedTraces()[neuron]
    signal -= signal.mean()
    signal /= signal.std()
    ax = layout.axes["example_{}".format(i+1)]["axis"]
    lw = matplotlib.rcParams["lines.linewidth"]*0.5
    fv = fancyViz.SwitchSchematicPlot(sess, linewidth=lw)
    fv.draw(signal, ax=ax)
    
layout.insert_figures('target_layer_name')
layout.write_svg(outputFolder / "staySwitch.svg")