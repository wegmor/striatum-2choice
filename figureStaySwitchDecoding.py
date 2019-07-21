#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:15:03 2019

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
import cmocean
plt.ioff()


#%%
style.set_context()

endoDataPath = pathlib.Path("data") / "endoData_2019.hdf"
outputFolder = pathlib.Path("svg")
cacheFolder =  pathlib.Path("cache")
templateFolder = pathlib.Path("templates")

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#%%
layout = figurefirst.FigureLayout(templateFolder / "staySwitchDecoding.svg")
layout.make_mplfigures()

#%%
M = pd.DataFrame() # confusion matrices (shuffle and real)
rP = pd.DataFrame() # action (probability) predictions for real data
sP = pd.DataFrame() # action (probability) predictions for shuffled data
C = pd.DataFrame() # svm coefficients

for action in ['mL2C']:#,'pC2L','mC2L','mR2C','pC2R','mC2R']:
    (rm,rp,rc), (sm,sp,sc) = analysisStaySwitchDecoding.decodeStaySwitch(endoDataPath, action)
    
    for df in [rm,rp,rc,sm,sp,sc]:
        df.insert(0, 'action', action)
    
    m = pd.concat([rm,sm], axis=0, keys=[False,True], names=['shuffled']).reset_index()
    M = M.append(m)
    
    rP = rP.append(rp)
    sP = sP.append(sp)

    c = pd.concat([rc,sc], axis=0, keys=[False,True], names=['shuffled']).reset_index()
    C = C.append(c)
    

#%%
rwAvg = (M.groupby(['genotype','true','predicted'])
          .apply(analysisStaySwitchDecoding.wAvg, 'percent_real', 'noNeurons'))
rsem = (M.groupby(['genotype','true','predicted'])
         .apply(analysisStaySwitchDecoding.bootstrap, 'percent_real', 'noNeurons'))
swAvg = (M.groupby(['genotype','true','predicted'])
          .apply(analysisStaySwitchDecoding.wAvg, 'percent_shuffle', 'noNeurons'))
ssem = (M.groupby(['genotype','true','predicted'])
         .apply(analysisStaySwitchDecoding.bootstrap, 'percent_shuffle', 'noNeurons'))

wAvg = pd.concat([rwAvg, swAvg], axis=1, keys=['real', 'shuffle'])
sem = pd.concat([rsem, ssem], axis=1, keys=['real', 'shuffle'])
cm_df = pd.concat([wAvg, sem], axis=1, keys=['mean','sem'])


#%%
for (g,a), df in cm_df.groupby(['genotype','action']):
    ax = layout.axes['cm_{}_{}'.format(g,a)]['axis']
    
    cm_mean = df.unstack('predicted')['mean','real']
    cm_sem = df.unstack('predicted')['sem','real']
    cm_annot = np.apply_along_axis(lambda p: '{:.0%}\n±{:.0%}'.format(*p),
                                   2, np.stack([cm_mean, cm_sem],
                                               axis=2))
    
    cmap = sns.light_palette(style.getColor(a), as_cmap=True, reverse=False)
    
    sns.heatmap(cm_mean, annot=cm_annot, fmt='', cmap=cmap, square=True,
                xticklabels='', yticklabels='', cbar=False, vmin=0, vmax=1, 
                annot_kws={'fontsize':6, 'ha':'center', 'va':'center'}, ax=ax)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    #ax.axis('off')


#%%
layout.insert_figures('plots')
layout.write_svg(outputFolder / "staySwitchDecoding.svg")