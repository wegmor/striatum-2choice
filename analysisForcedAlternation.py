import numpy as np
import pandas as pd
import tqdm
from utils import readSessions
from utils.cachedDataFrame import cachedDataFrame

def getActionAverages(traces, apf):
    keepLabels = ['pC2L-', 'mC2L-',
                  'pC2R-', 'mC2R-',
                  'pL2Cd', 'pL2Co', 'pL2Cr', 'mL2C-',
                  'pR2Cd', 'pR2Co', 'pR2Cr', 'mR2C-']
    apf = apf.loc[apf.label.isin(keepLabels)].copy()
    apf['label'] = apf.label.astype('str')
    actionAvg = traces.loc[apf.index].groupby([apf.label,apf.actionNo]).mean().dropna()
    labels = apf.groupby(['label','actionNo']).first().loc[actionAvg.index,
                                                           ['actionDuration']]    
    return(actionAvg, labels)

def wAvg(group, var, weights):
    return(np.average(group[var], weights=group[weights]))
    
def bootstrap(group, var, weights, iterations=1000):
    avgs = []
    for _ in range(iterations):
        idx = np.random.choice(len(group[var]), size=len(group[var]),
                               replace=True)
        avgs.append(np.average(group[var].iloc[idx], weights=group[weights].iloc[idx]))
    return(np.std(avgs))
    
def jitter(x, std):
    return(x+np.random.normal(0,std,size=len(x)))
    
    
#%%
@cachedDataFrame("forcedAlternationTunings.pkl")
def getTuningData(dataFilePath, no_shuffles=1000):
    df = pd.DataFrame()
    for s in readSessions.findSessions(dataFilePath, task='forcedAlternation'):
        traces = s.readDeconvolvedTraces(zScore=True).reset_index(drop=True) # frame no as index
        apf = s.labelFrameActions(switch=False, reward='sidePorts',
                                  splitCenter=True).reset_index(drop=True)
        
        # TODO: fix remaining recordings with dropped frames
        if traces.shape[0] != apf.shape[0]:
            continue
        
        actionAvg, labels = getActionAverages(traces, apf)  # mean per action
        actionAvg = actionAvg.groupby('label').mean()       # mean per label

        shuffle_labels = s.shuffleFrameLabels(no_shuffles, switch=False,
                                              reward='sidePorts', splitCenter=True)
        
        s_actionAvgs = []
        for s_apf in tqdm.tqdm(shuffle_labels, desc="{} (shuffling)".format(s)):
            s_actionAvg, s_labels = getActionAverages(traces, s_apf)
            s_actionAvg = s_actionAvg.groupby('label').mean()
            s_actionAvgs.append(s_actionAvg)
        s_actionAvgs = pd.concat(s_actionAvgs, keys=np.arange(no_shuffles),
                                 names=['shuffle']).reorder_levels(['label','shuffle'])

        for action, adata in tqdm.tqdm(s_actionAvgs.groupby('label'), desc=str(s)):
            for neuron in adata:
                ndict = {}
                dist = adata[neuron].values # shuffled "label" means distribution
                value = actionAvg.loc[action, neuron] # actual mean
                
                ndict['genotype'] = s.meta.genotype
                ndict['animal'] = s.meta.animal
                ndict['date'] = s.meta.date
                ndict['neuron'] = neuron
                ndict['action'] = action
                ndict['mean'] = value
                ndict['s_mean'] = dist.mean()
                ndict['s_std'] = dist.std()
                ndict['tuning'] = (ndict['mean'] - ndict['s_mean']) / ndict['s_std']
                # v percentile of the actual mean in the shuffled distribution
                ndict['pct'] = np.searchsorted(np.sort(dist), value) / len(dist)
                
                df = df.append(pd.Series(ndict), ignore_index=True)
        
    return df

#%%
def getFASessionStats(dataFilePath):
    # load sessions meta data, reduce to imaged sessions of animals that performed FA
    meta = pd.read_hdf(dataFilePath, 'meta')
    meta = meta.loc[(meta.caRecordings.str.len() != 0) &
                    (meta.task.isin(['2choice','forcedAlternation','2choiceAgain'])) &
                    (meta.animal.isin(meta.query('task == "forcedAlternation"').animal.unique()))
                   ]
    # sort index by date
    meta['date_fmt'] = pd.to_datetime(meta.date, yearfirst=True)
    meta = meta.set_index(['genotype','animal','date_fmt']).sort_index()
    # count recording sessions backwards from last -1 to first
    meta['noRecSessions'] = meta.groupby(['genotype','animal']).size()
    meta['recSession'] = np.concatenate([np.arange(-n,0) for n in 
                                         meta.groupby(['genotype','animal']).noRecSessions.first()])
    # keep final 9 sessions (5x 2choice, 3x FA, 1x 2choice)
    meta = meta[meta.recSession > -10].reset_index()
    
    sessionStats = pd.DataFrame()
    for m,s in [(m,readSessions.Session(dataFilePath, m)) for _,m in meta.iterrows()]:
        df = s.readSensorValues(slim=False)
        
        # side exits
        df['leftEx'] = df.beamL.diff() == -1
        df['rightEx'] = df.beamR.diff() == -1
        
        # reduce df to choice port exits
        df = df.loc[df.leftEx | df.rightEx,
                    ['animal','date','leftEx','rightEx',
                     'rewardNo','rewardP']].copy()
        
        # reward port switches prior to port exit
        df['rewardP'] = df.rewardP.shift(1)
        
        # define reward -- it is delivered when the beam is still broken,
        # after 350 ms delay, before port exit
        df['reward'] = (df.rewardNo.diff() >= 1).astype('bool')
        df['correct'] = ((df.leftEx & (df.rewardP == 'L')) |
                         (df.rightEx & (df.rewardP == 'R')))
    
        # convert to int
        df['leftEx'] = df.leftEx.astype('int')
        df['rightEx'] = df.rightEx.astype('int')
        df['correct'] = df.correct.astype('int')
        df['reward'] = df.reward.astype('int')
        
        # label switch trials
        df['switch'] = (df.leftEx.diff().abs() == 1).astype('int')
        
        # no of choice port exits
        df = df.dropna() # diffs/shift(1) lead to NaN in first row
        df['sideExNo'] = np.cumsum(df.leftEx | df.rightEx)
        df.set_index('sideExNo', inplace=True)
        
        # get stats
        df['bin'] = pd.cut(df.index, 3)
        df['bin'] = df.bin.cat.codes
        
        stats = df.groupby('bin')[['leftEx','rightEx','switch','correct','reward']].sum()
        stats['trials'] = stats.leftEx + stats.rightEx
        stats['genotype'] = m.genotype
        stats['animal'] = m.animal
        stats['session'] = m.recSession
        stats['task'] = m.task
      
        # append to df
        sessionStats = sessionStats.append(stats.reset_index(), ignore_index=True)
        
    return sessionStats