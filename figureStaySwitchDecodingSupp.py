#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:10:35 2019

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
import analysisStaySwitchDecoding
plt.ioff()


#%%
style.set_context()

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
cachedDataPath = cacheFolder / 'staySwitchAUC.pkl'
if cachedDataPath.is_file():
    staySwitchAUC = pd.read_pickle(cachedDataPath)
else:
    staySwitchAUC = analysisStaySwitchDecoding.getWStayLSwitchAUC(endoDataPath)
    staySwitchAUC.to_pickle(cachedDataPath)
   
    
#%%
staySwitchAUC['sign'] = (staySwitchAUC.pct > .995).astype('int') - (staySwitchAUC.pct < .005).astype('int')

for genotype in ['d1','a2a','oprm1']:
    layout = figurefirst.FigureLayout(templateFolder / "staySwitchDecodingSuppTuningAvgs.svg")
    layout.make_mplfigures()
    
    df = staySwitchAUC.query('genotype == @genotype & sign in [1,-1]').copy()
    
    for (gt,sign,action), pop_df in df.groupby(['genotype','sign','action']):
        axs = [layout.axes['{}{}_{}'.format(action,tt,{1:'stay',-1:'switch'}[sign])]['axis']
                   for tt in ['r.','o.','o!']]
        cax = layout.axes['colorbar']['axis']
        
        analysisStaySwitchDecoding.drawPopAverageFV('data/endoData_2019.hdf', pop_df, axs, cax)
        
    layout.insert_figures('plots')
    layout.write_svg(outputFolder / "staySwitchDecodingSuppTuningAvgs_{}.svg".format(genotype))
    
