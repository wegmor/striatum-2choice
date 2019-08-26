import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import h5py
import pathlib
import figurefirst

import analysisOpenField
import style

style.set_context()
plt.ioff()


#%%
endoDataPath = pathlib.Path('.') / "endoData_2019.hdf"
alignmentDataPath = pathlib.Path('.') / "alignment_190227.hdf"
outputFolder = pathlib.Path('svg')
cacheFolder = pathlib.Path('cache')
templateFolder = pathlib.Path('striatum_2choice/templates')

if not outputFolder.is_dir():
    outputFolder.mkdir()
if not cacheFolder.is_dir():
    cacheFolder.mkdir()

#All 4-behavior panels    
cachedDataPath = cacheFolder / "segmentedBehavior.pkl"
if cachedDataPath.is_file():
    segmentedBehavior = pd.read_pickle(cachedDataPath)
else:
    segmentedBehavior = analysisOpenField.segmentAllOpenField(endoDataPath)
    segmentedBehavior.to_pickle(cachedDataPath)

segmentedBehavior = segmentedBehavior.set_index("session")


cachedDataPath = cacheFolder / "openFieldDecodingWithIncreasingNumberOfNeurons.pkl"
if cachedDataPath.is_file():
    decodingData = pd.read_pickle(cachedDataPath)
else:
    decodingData = analysisOpenField.decodeWithIncreasingNumberOfNeurons(endoDataPath, segmentedBehavior)
    decodingData.to_pickle(cachedDataPath)

decodingData.insert(1, "genotype", decodingData.session.str.split("_").str[0])

#plt.sca(layout.axes["decodeWithIncreasingNumberOfNeurons"]["axis"])
plt.figure(figsize=(3,3))
for strSess, df in decodingData.groupby("session"):
    genotype = strSess.split("_")[0]
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=0.2, lw=.5)
    plt.plot(df.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
             alpha=0.2, lw=.5)
for genotype, df in decodingData.groupby("genotype"):
    plt.plot(df.groupby("nNeurons").realAccuracy.mean(), color=style.getColor(genotype),
             alpha=1.0)
plt.plot(decodingData.groupby("nNeurons").shuffledAccuracy.mean(), color=style.getColor("shuffled"),
         alpha=1.0)

order = ("oprm1", "d1", "a2a")
meanHandles = [matplotlib.lines.Line2D([], [], color=style.getColor(g)) for g in order]
shuffleHandle = matplotlib.lines.Line2D([], [], color=style.getColor("shuffled"))
plt.legend(meanHandles+[shuffleHandle], order+("shuffled",), loc=(0.45, 0.22), ncol=2)

plt.ylim(0,1)
plt.xlim(0,300)
plt.xlabel("Number of neurons")
plt.ylabel("Decoding accuracy (%)")
plt.yticks(np.linspace(0,1,5), np.linspace(0,100,5,dtype=np.int64))
sns.despine(ax=plt.gca())
plt.savefig("openFieldDecoding.svg", dpi=300)
