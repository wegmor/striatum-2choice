cimport numpy as cnp
import pandas as pd
import numpy as np

cdef enum:
    left=0
    center=1
    right=2

cdef tuple _findPrevEntry(cnp.int_t[:,:] beams, Py_ssize_t i):
    while i>0:
        if beams[i-1, left] == 1: return (u"L", i)
        if beams[i-1, center] == 1: return (u"C", i)
        if beams[i-1, right] == 1: return (u"R", i)
        i -= 1
    return (u"N", -1)
    
cdef list _findTrials(cnp.int_t[:,:] beams, cnp.int_t[:,:] leds, cnp.int_t[:] rewardNo,
                 cnp.ndarray rewardPort, cnp.ndarray time):
    cdef Py_ssize_t i=0, T = beams.shape[0]
    cdef cnp.int_t prevExit, enterCenter, exitCenter, enterSide, exitSide, reward
    cdef Py_UNICODE prevPort, chosenPort, rewardedPort
    cdef cnp.int_t successfulInit, successfulEnd
    cdef list trials = []
    
    #T is number of frames
    while i<T:
        
        #Find when center led was turned on
        #while i<T and leds[i, center] == 0: i += 1 #or beams[i, center] == 1
            
        #Find when entering center port
        while i<T and beams[i, center] == 0: i += 1
        enterCenter = i
        
        #Find when exiting center port
        while i<T and beams[i, center] == 1: i += 1
        exitCenter = i
        
        #Was the center led turned off?
        successfulInit = (i<T and leds[i, center] == 0)
        
        if successfulInit and (leds[i, left] == 0 or leds[i, right] == 0):
            raise Exception("LED assumption violated")
        
        #Until one port is entered
        while i<T:
            #Either side port
            if beams[i, left]==1 or beams[i, right]==1:
                
                #Which port is chosen?
                if beams[i, left]==1: chosenPort = u"L"
                else: chosenPort = u"R"
                enterSide = i
                
                #Find when port was exited
                if chosenPort == u"L":
                    while i<T and beams[i, left]==1: i+=1
                elif chosenPort == u"R":
                    while i<T and beams[i, right]==1: i+=1
                exitSide = i
                if i>=T: break
                    
                successfulEnd = (leds[i, center] == 1)
                
                #Any rewards?
                reward = rewardNo[exitSide] - rewardNo[enterSide]
                
                prevPort, prevExit = _findPrevEntry(beams, enterCenter)
                trials.append((prevPort, time[prevExit], time[enterCenter], time[exitCenter], successfulInit,
                               time[enterSide], time[exitSide], successfulEnd, chosenPort,
                               rewardPort[enterSide], reward))
                break
                
            #Center port
            if beams[i, center]==1:
                prevPort, prevExit = _findPrevEntry(beams, enterCenter)
                trials.append((prevPort, time[prevExit], time[enterCenter], time[exitCenter], successfulInit, -1, -1, -1, u"C", rewardPort[i], 0))
                break
                
            i += 1
    return trials

def findTrials(sensorValues, timeColumn="frameNo"):
    '''Estimate all trial attempts, including incorrectly initialized ones.
    
    Arguments:
    sensorValues --- A Pandas Dataframe as given by block.readSensorValues()
    timeColumn --- The column of sensorValues to use as time measurement. Recommended
                   options are "time" and "frameNo", depending on whether it will
                   later be matched with calcium recordings. Possibly other columns
                   could be used as well (trialNo?), but this is untested.
    
    Returns:
    A Pandas Dataframe where each row is a trial attempt.
    '''
    res = _findTrials(sensorValues[["beamL","beamC", "beamR"]].values,
                      sensorValues[["ledL","ledC", "ledR"]].values,
                      sensorValues.rewardNo.values,
                      sensorValues.rewardP.values,
                      sensorValues[timeColumn].values)
    return pd.DataFrame(res, columns=["previousPort", "exitPrevious", "enterCenter", "exitCenter",
                                      "successfulCenter", "enterSide", "exitSide",
                                      "successfulSide", "chosenPort", "rewardedPort", "reward"])


cdef tuple _labelFrameActions(cnp.int_t[:,:] beams, cnp.int_t[:] rewardNo,
                    cnp.int_t includeRewards=True, cnp.int_t includeSwitches=False):
    cdef Py_ssize_t i=0, T = beams.shape[0]
    cdef cnp.int_t[:] fromPort = np.zeros(T, np.int), toPort = np.zeros(T, np.int)
    cdef cnp.int_t[:] inPort = np.zeros(T, np.int)
    cdef cnp.int_t[:] lastEvent = np.zeros(T, np.int), nextEvent = np.zeros(T, np.int)
    cdef cnp.int_t[:] reward = np.zeros(T, np.int)
    cdef cnp.int_t[:] lastChoice = np.zeros(T, np.int), nextChoice = np.zeros(T, np.int)
    
    cdef object labels = np.zeros(T, object)
    cdef cnp.ndarray[cnp.int_t, ndim=1] actionDuration = np.zeros(T, np.int), actionNo = np.zeros(T, np.int)
    cdef cnp.ndarray[cnp.float_t, ndim=1] actionProgress = np.zeros(T, np.float)
    
    #Which, if any, port is the mouse in?
    for i in range(T):
        inPort[i] = -1
        for p in range(3):
            if beams[i, p]==1:
                inPort[i] = p
            
    #Forward pass
    fromPort[0] = -1
    lastChoice[0] = -1
    for i in range(1, T):
        if inPort[i-1] == inPort[i]:
            lastEvent[i] = lastEvent[i-1]
        else:
            lastEvent[i] = i
        if inPort[i-1] == -1 and inPort[i] != -1:
            fromPort[i] = inPort[i]
        else:
            fromPort[i] = fromPort[i-1]
        if inPort[i] == center:
            reward[i] = 0
        elif inPort[i] == left or inPort[i] == right:
            if rewardNo[i] > rewardNo[i-1]:
                reward[i] = 2
                lastEvent[i] = i
            elif i-lastEvent[i]>=7 and reward[i-1] == 0:
                reward[i] = 1
                lastEvent[i] = i
            elif inPort[i-1] == -1:
                reward[i] = 0
            else:
                reward[i] = reward[i-1]
        elif reward[i-1] == 0 and inPort[i] == -1 and (fromPort[i] == left or fromPort[i] == right):
            reward[i] = 1
        else:
            reward[i] = reward[i-1]
        if inPort[i-1] == -1 and (inPort[i] == left or inPort[i] == right):
            lastChoice[i] = inPort[i]
        else:
            lastChoice[i] = lastChoice[i-1]
            
    #Backward pass
    toPort[T-1] = -1
    nextEvent[T-1] = T-1
    nextChoice[T-1] = -1
    for i in range(T-2, -1, -1):
        if inPort[i] == inPort[i+1] and reward[i] == reward[i+1]:
            nextEvent[i] = nextEvent[i+1]
        else:
            nextEvent[i] = i
        if inPort[i] == -1 and inPort[i+1] != -1:
            toPort[i] = inPort[i+1]
        else:
            toPort[i] = toPort[i+1]
        if inPort[i] == -1 and (inPort[i+1] == left or inPort[i+1] == right):
            nextChoice[i] = inPort[i+1]
        else:
            nextChoice[i] = nextChoice[i+1]
            
    #Create labels
    portCodes = "-LCR"
    for i in range(T):
        if inPort[i] == -1:
            if nextEvent[i] - lastEvent[i] < 30:
                labels[i] = "m"
            else:
                labels[i] = "u"
        else:
            labels[i] = "p"
        labels[i] += portCodes[fromPort[i]+1]
        labels[i] += "2"
        labels[i] += portCodes[toPort[i]+1]
        if includeRewards == 1:
            labels[i] += "-or"[reward[i]]
        elif includeRewards == 2:
            if inPort[i] == left or inPort[i] == right:
                labels[i] += "-or"[reward[i]]
            else:
                labels[i] += "-"
            
        if includeSwitches:
            if nextChoice[i] == lastChoice[i] or nextChoice[i] == -1:
                labels[i] += "."
            else:
                labels[i] += "!"
        actionDuration[i] = nextEvent[i] - lastEvent[i] + 1
        actionProgress[i] = (i - lastEvent[i]) / float(actionDuration[i])
        if i>0:
            if labels[i] == labels[i-1]:
                actionNo[i] = actionNo[i-1]
            else:
                actionNo[i] = actionNo[i-1] + 1
    
    return labels, actionNo, actionProgress, actionDuration

def labelFrameActions(sensorValues, includeRewards=True, includeSwitches=False):
    '''Assign a string code to every frame, indicating where in the task the mouse currently is.
    
    Arguments:
    sensorValues --- A Pandas Dataframe as given by block.readSensorValues()
    includeRewards --- Whether to indicate rewards / omissions in the codes. Can be
                       True: Include for ports and return movements. "ports": Include only for ports.
                       False: Never include rewards.
    includeSwitches -- Whether to indicate stay / switch trials. Stay is indicated by "." and switches
                       by "!".   
    '''
    beams = sensorValues[["beamL","beamC", "beamR"]].values
    rewardNo = sensorValues["rewardNo"].values
    if includeRewards == "ports": includeRewards = 2
    labels, actionNo, actionProgress, actionDuration = _labelFrameActions(beams, rewardNo, includeRewards, includeSwitches)
    columns = {"label": labels, "actionNo": actionNo,
               "actionProgress": actionProgress, "actionDuration": actionDuration}
    return pd.DataFrame(columns)