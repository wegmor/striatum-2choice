#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:46:07 2019

@author: mowe
"""

import pandas as pd
from utils import readSessions


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
        sensor['delay'] = sensor.label.str.endswith('d').astype('int')
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


