import collections

import numpy as np
import pandas as pd
import tqdm
import scipy.spatial

from utils import readSessions, particleFilter
from utils.cachedDataFrame import cachedDataFrame

import analysisOpenField

def find2choiceSessionsFollowingOpenField(dataFile):
    for of_sess in readSessions.findSessions(dataFile, task="openField"):
        if of_sess.meta.date == '190224':
            #These don't have a 2-choice session before, so ignore
            continue
        sess = next(readSessions.findSessions(dataFile, task="2choice",
                                              animal=of_sess.meta.animal,
                                              date=of_sess.meta.date))
        yield sess

@cachedDataFrame("filteredTwoChoice.pkl")
def filterAllTwoChoice(dataFile):
    '''Run (and cache) particle filter on all 2-choice sessions associated
    with open field sessions.'''
    all_filtered = []
    for sess in find2choiceSessionsFollowingOpenField(dataFile):
        deconv = sess.readDeconvolvedTraces(indicateBlocks=True)
        tracking = sess.readTracking(inCm=True)
        if len(deconv) != len(tracking):
            continue
        tracking.index = deconv.index
        blocks = tracking.index.levels[0]
        filtered = []
        for block in blocks:
            t = tracking.loc[block]
            filtered.append(particleFilter.particleFilter(t, nParticles=2000, flattening=1e-12,
                                                          tqdmKwargs={'desc': str(sess)}))
        filtered = pd.concat(filtered)
        filtered.rename(columns={"bodyAngle": "bodyDirection"}, inplace=True)
        filtered.rename_axis("time", axis=0, inplace=True)
        filtered.bodyDirection *= 180/np.pi
        ind = tracking.index.to_frame()
        ind.insert(0, "session", str(sess))
        filtered.index = pd.MultiIndex.from_frame(ind)
        all_filtered.append(filtered)
    return pd.concat(all_filtered)

def twoChoiceSegmentKinematics(dataFile):
    kinematicParams = ["bodyAngleSpeed", "speed", "elongation"]
    filtered_all = filterAllTwoChoice(dataFile)
    avg_all = []
    for sess in find2choiceSessionsFollowingOpenField(dataFile):
        deconv = sess.readDeconvolvedTraces(indicateBlocks=True)
        lfa = sess.labelFrameActions(reward="never")
        if len(deconv) != len(lfa):
            continue
        filtered = filtered_all.loc[str(sess)]
        filtered = filtered.reset_index(drop=True)[kinematicParams]
        avg = filtered[kinematicParams].groupby(lfa.actionNo).mean()
        avg["label"] = lfa.groupby("actionNo").label.first()
        ind = avg.index.to_frame()
        ind.insert(0, "session", str(sess))
        avg.index = pd.MultiIndex.from_frame(ind)
        avg_all.append(avg)
    avg_all = pd.concat(avg_all)
    return avg_all

def openFieldSegmentKinematics(dataFile):
    kinematicParams = ["bodyAngleSpeed", "speed", "elongation"]
    
    filtered = analysisOpenField.filterAllOpenField(dataFile)
    segmented = analysisOpenField.segmentAllOpenField(dataFile)
    
    reindexed = []
    for s, g in segmented.groupby(level=0):
        fullIndex = np.arange(g.startFrame.iloc[0], g.stopFrame.iloc[-1])
        g = g.reset_index().set_index("startFrame").reindex(fullIndex, method="ffill")
        for param in kinematicParams:
            g[param] = filtered[param].loc[s].values
        reindexed.append(g)
    reindexed = pd.concat(reindexed)
    #reindexed = reindexed.reset_index().set_index(["session", "startFrame"])
    reindexed = reindexed[["session", "actionNo"] + kinematicParams]
    grouped = reindexed.groupby(["session", "actionNo"]).mean()
    grouped["label"] = segmented.reset_index("block").behavior
    return grouped

@cachedDataFrame("twoChoicePdists.pkl")
def twoChoicePdists(dataFile):
    kinematics_all = twoChoiceSegmentKinematics(dataFile)
    pdists_all = []
    for sess in tqdm.tqdm(find2choiceSessionsFollowingOpenField(dataFile)):
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        lfa = sess.labelFrameActions(reward="never")
        if len(deconv) != len(lfa):
            continue
        deconv_avg = deconv.groupby(lfa.actionNo).mean()
        kinematics = kinematics_all.loc[str(sess)]
        mask = np.logical_not(kinematics.label.str.startswith("u"))
        kinematics = kinematics[mask]
        deconv_avg = deconv_avg[mask]
        kinematics.drop("label", axis=1, inplace=True)
        kinematics_pdists = scipy.spatial.distance.pdist(kinematics, "mahalanobis")
        deconv_pdists = 1-scipy.spatial.distance.pdist(deconv_avg, "correlation")
        pdists = pd.DataFrame({'kinematics_dist': kinematics_pdists,
                               'deconv_dist': deconv_pdists})
        pdists["session"] = str(sess)
        pdists.set_index("session", drop=True, inplace=True)
        pdists_all.append(pdists)
    pdists_all = pd.concat(pdists_all)
    return pdists_all

@cachedDataFrame("openFieldPdists.pkl")
def openFieldPdists(dataFile):
    kinematics_all = openFieldSegmentKinematics(dataFile)
    segmented_all = analysisOpenField.segmentAllOpenField(dataFile)
    pdists_all = []
    for sess in readSessions.findSessions(dataFile, task="openField"):
        if sess.meta.date == '190224':
            #These don't have a 2-choice session before, so ignore
            continue
        deconv = sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        kinematics = kinematics_all.loc[str(sess)]
        fullRange = np.arange(segmented_all.loc[str(sess)].startFrame.iloc[0],
                              segmented_all.loc[str(sess)].stopFrame.iloc[-1])
        reindexed = segmented_all.loc[str(sess)].reset_index().set_index("startFrame").reindex(fullRange, method="ffill")
        deconv_avg = deconv.groupby(reindexed.actionNo).mean()

        kinematics.drop("label", axis=1, inplace=True)
        kinematics_pdists = scipy.spatial.distance.pdist(kinematics, "mahalanobis")
        deconv_pdists = 1-scipy.spatial.distance.pdist(deconv_avg, "correlation")
        pdists = pd.DataFrame({'kinematics_dist': kinematics_pdists,
                               'deconv_dist': deconv_pdists})
        pdists["session"] = str(sess)
        pdists.set_index("session", drop=True, inplace=True)
        pdists_all.append(pdists)
    pdists_all = pd.concat(pdists_all)
    return pdists_all

@cachedDataFrame("openFieldToTwoChoiceCdists.pkl")
def openFieldToTwoChoiceCdists(dataFile):
    kinematics_of_all = openFieldSegmentKinematics(dataFile)
    kinematics_tc_all = twoChoiceSegmentKinematics(dataFile)
    segmented_all = analysisOpenField.segmentAllOpenField(dataFile)
    cdists_all = []
    for of_sess in readSessions.findSessions(dataFile, task="openField"):
        if of_sess.meta.date == '190224':
            continue
        
        #Open field
        deconv_of = of_sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        kinematics_of = kinematics_of_all.loc[str(of_sess)]
        fullRange = np.arange(segmented_all.loc[str(of_sess)].startFrame.iloc[0],
                              segmented_all.loc[str(of_sess)].stopFrame.iloc[-1])
        reindexed = segmented_all.loc[str(of_sess)].reset_index().set_index("startFrame").reindex(fullRange, method="ffill")
        deconv_of_avg = deconv_of.groupby(reindexed.actionNo).mean()
        kinematics_of.drop("label", axis=1, inplace=True)
        
        #Two choice
        tc_sess = next(readSessions.findSessions(dataFile, task="2choice",
                                                 animal=of_sess.meta.animal,
                                                 date=of_sess.meta.date))
        deconv_tc = tc_sess.readDeconvolvedTraces(rScore=True).reset_index(drop=True)
        lfa = tc_sess.labelFrameActions(reward="never")
        if len(deconv_tc) != len(lfa):
            continue
        deconv_tc_avg = deconv_tc.groupby(lfa.actionNo).mean()
        kinematics_tc = kinematics_tc_all.loc[str(tc_sess)]
        mask = np.logical_not(kinematics_tc.label.str.startswith("u"))
        kinematics_tc = kinematics_tc[mask]
        deconv_tc_avg = deconv_tc_avg[mask]
        kinematics_tc.drop("label", axis=1, inplace=True)
        
        kinematics_cdists = scipy.spatial.distance.cdist(kinematics_of,
                                                         kinematics_tc,
                                                         "mahalanobis")
        deconv_cdists = 1-scipy.spatial.distance.cdist(deconv_of_avg, deconv_tc_avg,
                                                       "correlation")
        
        cdists = pd.DataFrame({'kinematics_dist': kinematics_cdists.flat,
                               'deconv_dist': deconv_cdists.flat})
        cdists["session"] = str(of_sess)
        cdists.set_index("session", drop=True, inplace=True)
        cdists_all.append(cdists)
    cdists_all = pd.concat(cdists_all)
    return cdists_all
