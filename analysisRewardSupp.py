#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:46:07 2019

@author: mowe
"""

import pandas as pd
import numpy as np
import tqdm
from utils import readSessions, cachedDataFrame
    
    
#%%
def getRewardRespData(endoDataPath, tuningData):
    def insertSessInfo(df, genotype, animal, date):
        for (k,v) in [('animal',animal),('genotype',genotype),('date',date)]:
            df.insert(0, k, v)
        
    lt_avg_trial = pd.DataFrame()
    rt_avg_trial = pd.DataFrame()
    lt_avg_frame = pd.DataFrame()
    rt_avg_frame = pd.DataFrame()
    
    for (genotype, animal, date), ts in tuningData.groupby(['genotype','animal','date']):
        ts.set_index(['neuron','action'], inplace=True)
        ts = ts[['signp','signn']].unstack('action') # 1 row/neuron; 1 column/tuning
    
        # load traces, labels, and sensor data
        s = next(readSessions.findSessions(endoDataPath, genotype=genotype, animal=animal,
                                           date=date, task='2choice'))
        deconv = s.readDeconvolvedTraces(zScore=True)
        lfa = s.labelFrameActions(reward='sidePorts')
        sensor = s.readSensorValues().reset_index(drop=True)
        
        if not len(deconv) == len(lfa) == len(sensor):
            continue
        
        # define no. rewards delivered & port occupied for every action
        sensor['actionNo'] = lfa.actionNo
        sensor['label'] = lfa.label
        sensor['reward'] = sensor.rewardNo.diff().fillna(0).astype('int')
        sensor.loc[~sensor.label.str.contains('[or]$'), 'reward'] = -1 # 0+: only omissions/reward deliveries
        sensor.loc[sensor.label.str.contains('[or]$'), 'actionNo'] -= 1 # r/o have same actionNo as preceding d
        sensor['actionFrame'] = sensor.groupby('actionNo').cumcount()
        sensor['delay'] = sensor.label.str.endswith('d').astype('int') # TODO: BUG, this isn't how the labeling works anymore!
        sensor['side'] = (sensor.beamL+sensor.beamC*2+sensor.beamR*3).replace({1:'L',2:'C',3:'R',0:'-'})
        sensor.set_index('actionNo', inplace=True)
        sensor['reward'] = sensor.groupby('actionNo').reward.max() # reward indicated for each frame of action,
                                                                   # delay only is -1
        deconv.index = sensor.reset_index().set_index(['side','reward','delay','actionNo','actionFrame']).index
        deconv = deconv.query('side in ["L","R"] & reward >= 0').copy()
        delayAndOutcomeANs = (deconv.reset_index().groupby('actionNo').delay.nunique() == 2).index.values
        deconv = deconv.query('actionNo in @delayAndOutcomeANs').copy() # only trials with delay and outcome
        
        # select neurons tuned to reward but not to omissions
        lt_deconv = deconv.iloc[:,(ts['signp','pL2Cr'] & ~ts['signp','pL2Co']).values]
        rt_deconv = deconv.iloc[:,(ts['signp','pR2Cr'] & ~ts['signp','pR2Co']).values]
        
        if not lt_deconv.empty:
            # avg trial responses
            lt_avg_resp = lt_deconv.query('delay == 0').groupby(['side','reward','actionNo']).mean()
            lt_avg_resp = lt_avg_resp.groupby(['side','reward']).mean()
            lt_avg_resp.columns.name = 'neuron'
            lt_avg_resp = pd.DataFrame(lt_avg_resp.stack(), columns=['avg'])
            insertSessInfo(lt_avg_resp, genotype, animal, date)
            lt_avg_trial = lt_avg_trial.append(lt_avg_resp.reset_index(), ignore_index=True)
    
            # frame-by-frame average traces
            lt_avg_trace = lt_deconv.groupby(['side','reward','actionFrame']).mean()
            lt_avg_trace.columns.name = 'neuron'
            lt_avg_trace = pd.DataFrame(lt_avg_trace.stack(), columns=['avg'])
            insertSessInfo(lt_avg_trace, genotype, animal, date)
            lt_avg_frame = lt_avg_frame.append(lt_avg_trace.reset_index(), ignore_index=True)
        
        if not rt_deconv.empty:
            # avg trial responses
            rt_avg_resp = rt_deconv.query('delay == 0').groupby(['side','reward','actionNo']).mean()
            rt_avg_resp = rt_avg_resp.groupby(['side','reward']).mean()
            rt_avg_resp.columns.name = 'neuron'
            rt_avg_resp = pd.DataFrame(rt_avg_resp.stack(), columns=['avg'])
            insertSessInfo(rt_avg_resp, genotype, animal, date)
            rt_avg_trial = rt_avg_trial.append(rt_avg_resp.reset_index(), ignore_index=True)
            
            # frame-by-frame average traces
            rt_avg_trace = rt_deconv.groupby(['side','reward','actionFrame']).mean()
            rt_avg_trace.columns.name = 'neuron'
            rt_avg_trace = pd.DataFrame(rt_avg_trace.stack(), columns=['avg'])
            insertSessInfo(rt_avg_trace, genotype, animal, date)
            rt_avg_frame = rt_avg_frame.append(rt_avg_trace.reset_index(), ignore_index=True) 
   
    return(lt_avg_trial, rt_avg_trial, lt_avg_frame, rt_avg_frame)


#%%
@cachedDataFrame.cachedDataFrame('outcomeResponseData.pkl')
def getOutcomeResponseData(endoDataPath):
    # TODO: not how we do this
    staySwitchAUC = pd.read_pickle('cache/staySwitchAUC.pkl').set_index(['genotype','animal','date','action','neuron']).sort_index()
    actionValues = pd.read_pickle('cache/actionValues.pkl').set_index(['genotype','animal','date','actionNo']).sort_index()
    
    def insertSessInfo(df, sess):
        for (k,v) in [('animal',sess.meta.animal),('genotype',sess.meta.genotype),('date',sess.meta.date)]:
            df.insert(0, k, v)
    
    TRIALS = []
    TUNING = []
    DECONV = []
    for sess in tqdm.tqdm(readSessions.findSessions(endoDataPath, task='2choice')):
        # load data
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        labels = sess.labelFrameActions(switch=True, reward='fullTrial').reset_index(drop=True)
        sensors = sess.readSensorValues().reset_index(drop=True)
        try:
            aucs = staySwitchAUC.loc[sess.meta.genotype, sess.meta.animal, sess.meta.date]
            avs = actionValues.loc[sess.meta.genotype, sess.meta.animal, sess.meta.date]
        except KeyError:
            print('no AUCs/AVs available for session, skip!')
            continue
        
        # make sure number of frames matches
        if not len(deconv) == len(labels) == len(sensors):
            print('something is fucked, skip!')
            continue
        
        # generate trial meta-data dataframe
        labels.set_index([labels.actionNo, labels.actionProgress], inplace=True)
        sensors.set_index([labels.actionNo, labels.actionProgress], inplace=True)
        
        trialsDf = labels.groupby(level=0).first()[['label']].copy()
        trialsDf = trialsDf.loc[trialsDf.label.str.contains('^p[LR]2.[ro].')].copy()
        trialsDf['rSize'] = sensors.rewardNo.diff().groupby('actionNo').first()
        trialsDf['prevRSize'] = trialsDf.rSize.shift(1)
        trialsDf['value'] = avs.value
        trialsDf['prevValue'] = trialsDf.value.shift(1)
        trialsDf['side'] = trialsDf.label.str[1]
        trialsDf['prevSide'] = trialsDf.side.shift(1)
        trialsDf.set_index([list(map(lambda an: str(sess)+'_{0:d}'.format(an), trialsDf.index.values.astype('int')))],
                           inplace=True)
        trialsDf.index.name = 'trial'
        insertSessInfo(trialsDf, sess)
        TRIALS.append(trialsDf.reset_index().set_index(['genotype','animal','date']))
        
        # compute reward and delay tunings (lose-switch:-1, win-stay:1, untuned:0)
        lRew = (((aucs.loc['pL2C','pct'] > .995) & (aucs.loc['pL2C','auc'] > 0))*1 - \
                ((aucs.loc['pL2C','pct'] < .005) & (aucs.loc['pL2C','auc'] < 0))*1)
        rRew = (((aucs.loc['pR2C','pct'] > .995) & (aucs.loc['pR2C','auc'] > 0))*1 - \
                ((aucs.loc['pR2C','pct'] < .005) & (aucs.loc['pR2C','auc'] < 0))*1)
        lDel = (((aucs.loc['dL2C','pct'] > .995) & (aucs.loc['dL2C','auc'] > 0))*1 - \
                ((aucs.loc['dL2C','pct'] < .005) & (aucs.loc['dL2C','auc'] < 0))*1)
        rDel = (((aucs.loc['dR2C','pct'] > .995) & (aucs.loc['dR2C','auc'] > 0))*1 - \
                ((aucs.loc['dR2C','pct'] < .005) & (aucs.loc['dR2C','auc'] < 0))*1)
        tuningDf = pd.concat([lRew, rRew, lDel, rDel], keys=['leftReward','rightReward','leftDelay','rightDelay'], axis=1)
        tuningDf.set_index([list(map(lambda n: str(sess)+'_{0:d}'.format(n), tuningDf.index.values.astype('int')))],
                           inplace=True)
        tuningDf.index.name = 'neuron'
        insertSessInfo(tuningDf, sess)
        TUNING.append(tuningDf.reset_index().set_index(['genotype','animal','date']))
        
        # gernerate (#trials * #neurons) x #frames data frame
        deconvDf = deconv.set_index([labels.actionNo, labels.label]).reset_index().copy()
        
        deconvDf = deconvDf.loc[deconvDf.label.str.contains('^[dp][LR]2.[ro].')].copy()
        deconvDf.loc[deconvDf.label.str.contains('^d[LR]2.[ro].'), 'actionNo'] += 1
        deconvDf['actionFrame'] = deconvDf.groupby('actionNo').cumcount()
        deconvDf.set_index(['actionNo','actionFrame'], inplace=True)
        deconvDf.drop(columns=['label'], inplace=True)
        deconvDf = deconvDf.stack().unstack('actionFrame')
        deconvDf.index.names = deconvDf.index.names[:-1] + ['neuron']
        deconvDf.set_index([list(map(lambda an: str(sess)+'_{0:d}'.format(an), deconvDf.index.get_level_values(0).astype('int'))),
                            list(map(lambda n: str(sess)+'_{0:d}'.format(n), deconvDf.index.get_level_values(1).astype('int')))],
                           inplace=True)
        deconvDf.index.names = ['trial','neuron']
        DECONV.append(deconvDf.iloc[:,:47])
    
    TRIALS = pd.concat(TRIALS)
    TUNING = pd.concat(TUNING)
    DECONV = pd.concat(DECONV)

    return TRIALS, TUNING, DECONV

