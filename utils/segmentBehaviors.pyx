import numpy as np
import pandas as pd

cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs
cdef np.float_t dirDiff(np.float_t A, np.float_t B):
    cdef np.float_t a = (B-A+720)%360
    cdef np.float_t b = (A-B+720)%360
    if a < b:
        return a
    else:
        return -b
    
cdef np.float_t score_left_turn(np.float_t angle, np.float_t distance, Py_ssize_t duration):
    if duration < 5: return -10
    if angle < duration: return -10 #A factor of 1 happens to be good
    if distance > 10*duration: return -10
    return angle - 0.5*duration - 10

cdef np.float_t score_right_turn(np.float_t angle, np.float_t distance, Py_ssize_t duration):
    return score_left_turn(-angle, distance, duration)

cdef np.float_t score_running(np.float_t angle, np.float_t distance, Py_ssize_t duration):
    if duration < 5: return -10
    if angle > duration: return -10
    if -angle > duration: return -10
    return distance*5.0 - 10

cdef np.float_t score_stationary(np.float_t angle, np.float_t distance, Py_ssize_t duration):
    if duration*10 < distance: return -5
    return duration - 5

@cython.wraparound(False)
@cython.boundscheck(False)
cdef list optimize(np.float_t[:] x, np.float_t[:] y, np.float_t[:] bodyDirection):
    cdef Py_ssize_t N = x.shape[0]
    cdef np.float_t[:] cumulScore = np.zeros(N)
    cdef np.float_t[:] netTurn = np.zeros(N)
    cdef np.float_t[:] distances = np.zeros(N)
    cdef np.float_t[:,:] scores = np.zeros((N,4))
    cdef np.int_t[:] choice = np.zeros(N, dtype=np.int)
    cdef np.int_t[:] prev = np.zeros(N, dtype=np.int)
    cdef Py_ssize_t i, j, k, l
    cdef np.float_t angle, dx, dy, distance
    cdef np.float_t[4] candidates
    for i in range(N):
        angle = 0
        for j in range(i+1, min(N, i+2000)):
            angle += dirDiff(bodyDirection[j-1], bodyDirection[j])
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            distance = sqrt(dx*dx + dy*dy)
            candidates[0] = score_left_turn(angle, distance, j-i)
            candidates[1] = score_right_turn(angle, distance, j-i)
            candidates[2] = score_running(angle, distance, j-i)
            candidates[3] = score_stationary(angle, distance, j-i)
            for k in range(4):
                if candidates[k] + cumulScore[i] > cumulScore[j]:
                    cumulScore[j] = candidates[k] + cumulScore[i]
                    choice[j] = k
                    prev[j] = i
                    netTurn[j] = angle
                    distances[j] = distance
                    for l in range(4):
                        scores[j, l] = candidates[l]
    seq = list()
    i = N-1
    while i > 0:
        seq.append((prev[i],i,choice[i], netTurn[i], distances[i],
                    cumulScore[i], scores[i,0], scores[i,1], scores[i,2], scores[i,3]))
        i = prev[i]
    return seq[::-1]

def segmentBehaviors(tracking):
    '''Given a sequence of positions and directions, find likely behavioral segments.
    
    Arguments:
    tracking -- A pandas dataframe with at least the columns "x", "y" and "bodyDirection",
                and time as the index
                
    Returns:
    A pandas dataframe where each row corresponds to one  behavioral segment. The four
    possible categories are "rightTurn", "leftTurn", "running" and "stationary".
    '''
    for k in ('x', 'y', 'bodyDirection'):
        if k not in tracking:
            raise Exception(k + " not found in dataframe")
        if tracking[k].shape != (len(tracking.x),):
            raise Exception(k + " has the wrong dimension")
            
    res = optimize(tracking.x.values, tracking.y.values, tracking.bodyDirection.values)
    columns = ['startFrame', 'stopFrame', 'behavior', 'netTurn', 'distance',
                'cumulativeScore', 'leftScore', 'rightScore', 'runningScore', 'stationaryScore']
    res = pd.DataFrame(res, columns=columns)
    
    #Categorical behaviors
    behaviorNames = ['leftTurn', 'rightTurn', 'running', 'stationary']
    res.behavior = pd.Categorical.from_codes(res.behavior, behaviorNames, ordered=True)
    
    #Some more statistics
    res.insert(0, 'startTime', tracking.reset_index().time[res.startFrame].values)
    res.insert(1, 'stopTime', tracking.reset_index().time[res.stopFrame].values)
    res.insert(2, 'duration', res.stopTime - res.startTime)
    res.insert(5, 'numFrames', res.stopFrame - res.startFrame)
    
    #Fix an offset error
    res.stopFrame.iloc[-1] += 1
    return res
