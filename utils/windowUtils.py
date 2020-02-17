import numpy as np
import pandas as pd

def extendedLfa(lfa):
    nextLabel = lfa.groupby(lfa.actionNo-1).label.first().rename("nextLabel")
    prevLabel = lfa.groupby(lfa.actionNo+1).label.first().rename("prevLabel")
    lfa = lfa.join(nextLabel, "actionNo", "left")
    lfa = lfa.join(prevLabel, "actionNo", "left")
    return lfa

def binAroundAction(lfa, actionFilter, binsPerAction=5, includePrev=True, includeNext=True):
    if isinstance(actionFilter, str):
        mask = lfa.label == actionFilter
    elif isinstance(actionFilter, (np.ndarray, pd.Series)):
        if actionFilter.dtype != "bool":
            raise ValueError("actionFilter not understood.")
        mask = actionFilter
    else:
        raise ValueError("Unknown type of actionFilter")
    lfa["mainAction"] = mask
    lfa["relProgress"] = np.nan
    offset = 0
    if includePrev:
        isPrev = lfa.groupby(lfa.actionNo-1).mainAction.any().rename("isPrev")
        lfa = lfa.join(isPrev, "actionNo", "left")
        lfa.loc[lfa.isPrev==True, "relProgress"] = lfa.loc[lfa.isPrev==True, "actionProgress"]
        offset += 1
    lfa.loc[lfa.mainAction, "relProgress"] = offset + lfa.loc[lfa.mainAction, "actionProgress"]
    offset += 1
    if includeNext:
        isNext = lfa.groupby(lfa.actionNo+1).mainAction.any().rename("isNext")
        lfa = lfa.join(isNext, "actionNo", "left")
        #lfa.prevAction.fillna(False, inplace=True)
        lfa.loc[lfa.isNext==True, "relProgress"] = offset+lfa.loc[lfa.isNext==True, "actionProgress"]
    lfa["bin"] = np.floor((lfa["relProgress"]*binsPerAction))
    return lfa

