cimport numpy as cnp
import pandas as pd

cdef enum:
    left=0
    center=1
    right=2

cdef list _findTrials(cnp.int_t[:,:] beams, cnp.int_t[:,:] leds, cnp.int_t[:] rewardNo,
                 cnp.ndarray rewardPort, cnp.ndarray time):
    cdef Py_ssize_t i=0, T = beams.shape[0]
    cdef cnp.int_t enterCenter, exitCenter, enterSide, exitSide, reward
    cdef Py_UNICODE chosenPort, rewardedPort
    cdef cnp.int_t successfulInit
    cdef list trials = []
    
    #T is number of frames
    while i<T:
        
        #Find when center led was turned on
        while i<T and (leds[i, center] == 0 or beams[i, center] == 1): i += 1
            
        #Find when entering center port
        while i<T and beams[i, center] == 0: i += 1
        enterCenter = i
        
        #Find when exiting center port
        while i<T and beams[i, center] == 1: i += 1
        exitCenter = i
        
        #Was the center led turned off?
        successfulInit = (i<T and leds[i, center] == 0)
        
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
                #Any rewards?
                reward = rewardNo[exitSide] - rewardNo[enterSide]
                trials.append((time[enterCenter], time[exitCenter], successfulInit, time[enterSide], time[exitSide], chosenPort, rewardPort[enterSide], reward))
                break
                
            #Center port
            if beams[i, center]==1:
                trials.append((time[enterCenter], time[exitCenter], successfulInit, -1, -1, u"C", rewardPort[i], 0))
                break
                
            i += 1
    return trials

def findTrials(sensorValues):
    res = _findTrials(sensorValues[["beamL","beamC", "beamR"]].values,
                      sensorValues[["ledL","ledC", "ledR"]].values,
                      sensorValues.rewardNo.values,
                      sensorValues.rewardP.values,
                      sensorValues.frameNo.values)
    return pd.DataFrame(res, columns=["enterCenter", "exitCenter", "successfulInit",
                                      "enterSide", "exitSide", "chosenPort", "rewardedPort", "reward"])