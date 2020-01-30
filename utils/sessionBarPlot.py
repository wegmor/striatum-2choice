import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bootstrapSEM(values, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(values), len(values), replace=True)
        avgs.append(np.average(values.iloc[idx], weights=weights.iloc[idx]))
    return np.std(avgs)

def sessionBarPlot(df, yCol, ax, colorFunc, genotypeCol="genotype", animalCol="animal",
                   dateCol="date", weightCol="noNeurons", weightScale=0.1,
                   genotypeOrder=("d1","a2a","oprm1"), barAlpha=0.3, orientation='vertical'):

    genotypeMeans = df.groupby(genotypeCol).apply(lambda df: np.average(df[yCol], weights=df[weightCol]))
    genotypeSEMs = df.groupby(genotypeCol).apply(lambda df: bootstrapSEM(df[yCol], weights=df[weightCol]))
    genotypeMeans = genotypeMeans.loc[list(genotypeOrder)]
    genotypeSEMs = genotypeSEMs.loc[list(genotypeOrder)]
    barColors = [colorFunc(gt) for gt in genotypeMeans.index]
    if orientation == 'vertical':
        ax.bar(np.arange(len(genotypeOrder)), genotypeMeans, 0.9, color=barColors,
               alpha=barAlpha, yerr=genotypeSEMs, clip_on=False)
    else:
        ax.barh(np.arange(len(genotypeOrder)), genotypeMeans, 0.9, color=barColors,
                alpha=barAlpha, xerr=genotypeSEMs, clip_on=False)

    perAnimal = df.sort_values(dateCol).drop_duplicates(animalCol)
    cumCount = perAnimal.groupby(genotypeCol).cumcount()
    nAnimalsPerGenotype = perAnimal.groupby(genotypeCol).size().reindex(perAnimal[genotypeCol])
    genotypeIndex = pd.Series(np.arange(len(genotypeOrder)), genotypeOrder).reindex(perAnimal[genotypeCol])
    xCoord = (cumCount.values / (nAnimalsPerGenotype.values-1))*0.6 + 0.2 - 0.5 + genotypeIndex.values
    xCoord = pd.Series(xCoord, perAnimal[animalCol]).reindex(df[animalCol])
    xCoord.index = df.index
    c = [colorFunc(gt) for gt in df[genotypeCol]]
    if orientation == 'vertical':
        ax.scatter(xCoord, df[yCol], df[weightCol]*weightScale, edgecolors=c, facecolor="none",
                   clip_on=False, lw=.5)
        ax.set_xticks(np.arange(len(genotypeOrder)))
        ax.set_xticklabels(genotypeOrder)
    else:
        ax.scatter(df[yCol], xCoord, df[weightCol]*weightScale, edgecolors=c, facecolor='none',
                   clip_on=False, lw=.5)
        ax.set_yticks(np.arange(len(genotypeOrder)))
        ax.set_yticklabels(genotypeOrder)