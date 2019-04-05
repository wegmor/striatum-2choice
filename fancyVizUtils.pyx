cimport numpy as cnp
cimport libc.math

import numpy as np
import pandas as pd
import scipy.stats

def taskSchematicCoordinates(apf, maxLen=9999999):
    '''
    Converts the list of action taken per frame into coordinates for plotting on top
    of the task schematic. 
    
    Arguments:
    apf -- A pandas Dataframe with the action per frame, as calculated
           by block.calcActionsPerFrame()
    
    Returns:
    A pandas Dataframe with the x and y coordinates in the schematic for each frame
    '''
    return pd.DataFrame(_taskSchematicCoordinates(apf.action.values, apf.port.values,
                                                  apf.frameNo.values, apf.actionStart.values,
                                                  apf.actionStop.values, apf.rewarded.values,
                                                  maxLen),
                        columns=["x","y"], index=apf.index)

cdef cnp.ndarray[cnp.float_t, ndim=2] _taskSchematicCoordinates(str[:] action,
                                                                str[:] port,
                                                                cnp.int_t[:] frameNo,
                                                                cnp.int_t[:] actionStart,
                                                                cnp.int_t[:] actionStop,
                                                                cnp.float_t[:] rewarded,
                                                                cnp.int_t maxLen = 9999999
                                                               ):
    cdef Py_ssize_t i, N = action.shape[0]
    cdef cnp.float_t duration, progress, x, y, normal_x, normal_y
    cdef cnp.ndarray[cnp.float_t, ndim=2] coordinates = np.nan*np.ones((N,2)) 
    for i in range(N):
        duration = actionStop[i] - actionStart[i]
        if duration==0: continue
        else: progress = (frameNo[i] - actionStart[i]) / duration
        if action[i] == "inPort":
            if port[i] == "L": x = -4
            elif port[i] == "C": x = 0
            elif port[i] == "R": x = 4
            
            if port[i] != "C":
                if rewarded[i]>0: x *= 0.92
                else: x *= 1.08
            y = 1.0 - 2.0*progress
            if port[i] == "C": y = -y
            coordinates[i,0] = x
            coordinates[i,1] = y
            
        elif action[i] == "centerToSide":
            if duration > maxLen: continue
            x = progress * 4.0
            if port[i] == "L": x = -x
            y = 2.0*progress - 1.0
            y = 2 - y*y
            coordinates[i,0] = x
            coordinates[i,1] = y
            
        elif action[i] == "sideToCenter":
            if duration > maxLen: continue
            x = 4.0 - progress * 4.0
            if port[i] == "L": x = -x
            y = 2.0*progress - 1.0
            y = -2 + y*y
            normal_x = 2.0*(2.0*progress - 1.0)
            normal_y = 4.0
            normal_len = libc.math.sqrt(normal_x*normal_x + normal_y*normal_y)
            if port[i] == "L": normal_x = -normal_x
            if rewarded[i] <= 0:
                normal_x *= -1
                normal_y *= -1
            x += normal_x / normal_len / 3.0
            y += normal_y / normal_len / 3.0
            coordinates[i,0] = x
            coordinates[i,1] = y
            
    return coordinates

cpdef cnp.ndarray[cnp.float_t, ndim=2] integerHistogram(cnp.float_t[:,:] coords, cnp.float_t[:] weights,
                                                        Py_ssize_t width, Py_ssize_t height):
    '''
    Given a list of coordinates and a list of weights, bin the weights using the integer part
    of each coordinate.
    
    Arguments:
    coords -- A Nx2 numpy array of coordinates
    weights -- A length N numpy array with the weight of each point. Typically the 
               neural trace.
    width -- The width of the canvas in pixels
    height -- The height of the canvas in pixels 
    
    Returns:
    A numpy array where each element is the sum of the weights with those integer coordinates
    '''
    cdef cnp.ndarray[cnp.float_t, ndim=2] canvas = np.zeros((height, width))
    cdef Py_ssize_t i, N, r, c
    N = coords.shape[0]
    if weights.shape[0] != N:
        raise ValueError("Coordinates and weights have different lengths.")
    for i in range(0,N):
        if not libc.math.isnan(coords[i,0]) and not libc.math.isnan(coords[i,1]):
            c = <Py_ssize_t>coords[i,0]
            r = <Py_ssize_t>coords[i,1]
            canvas[r, c] += weights[i]
    return canvas

def returnBoutsCoordinates(returnBouts, sessionLength):
    '''
    Given a dataframe with start and stop of each return bout, find the plot coordinates
    of each frame.
    
    Arguments:
    returnBouts -- A pandas Dataframe with return bouts as given by `findReturnBouts`
    
    Returns:
    A pandas Dataframe with the x and y coordinates in the return-to-task plot for each frame
    '''
    cdef cnp.ndarray[cnp.float_t, ndim=2] coords = _returnBoutsCoordinates(
                                     returnBouts.start.values, returnBouts.stop.values,
                                     returnBouts.port.values, sessionLength)
    return pd.DataFrame(coords, columns=["x","y"])

cdef cnp.ndarray[cnp.float_t, ndim=2] _returnBoutsCoordinates(cnp.int_t[:] start, cnp.int_t[:] stop,
                                                              str[:] port, Py_ssize_t N):
    cdef Py_ssize_t i, j
    cdef cnp.float_t duration, progress, x, y
    cdef cnp.ndarray[cnp.float_t, ndim=2] coordinates = np.nan*np.ones((N,2))
    for j in range(start.shape[0]):
        duration = stop[j] - start[j]
        for i in range(start[j], stop[j]):
            progress = (i-start[j]) / duration
            coordinates[i,0] = progress * 50
            if port[j]=="L": coordinates[i,1] = 20
            elif port[j]=="C": coordinates[i,1] = 40
            elif port[j]=="R": coordinates[i,1] = 60
    return coordinates

def findBlockActions(sensorValues, timeout=40):
    '''
    Finds relevant parts of blocks: starts and stops of each trial, end of block by
    disengaging with the task and by switching to the other side. Start and stop of a
    trial are measured from the end of the previous trial to the exit of the side port.
    Non-task episodes are measured from the last exit of either a side port or the center
    port to the next entry into the center port, and occur when the time outside of any port
    exceeds `timeout` number of frames. Switches are measured starting from the exit of
    the side port and endind at the entry of the opposite side port (they thus typically
    include a center port visit in the middle).
    
    Arguments:
    sensorValues -- A pandas Dataframe containing the infrared beam readings. Can be obtained
                    by calling `block.readSensorValues()`.
    timeout -- The longest time the mouse is allowed to go without breaking any infrared beam,
               in number of frames. Longer absences than this will count as being disengaged.
    
    Returns:
    A pandas Dataframe where each row is one block-relevant action, as described above.
    '''
    return pd.DataFrame(_findBlockActions(sensorValues.beamL.values,
                                         sensorValues.beamC.values,
                                         sensorValues.beamR.values,
                                         timeout),
                        columns=["action", "port", "trialNo", "start", "stop"])

cdef list _findBlockActions(cnp.int_t[:] beamL, cnp.int_t[:] beamC, cnp.int_t[:] beamR,
                            cnp.int_t timeout=40):
    cdef Py_ssize_t i=0, j, N=beamL.shape[0]
    cdef int trialNum=0, currentBlock=0, lastExit=-9999, lastEnd=-9999, nonTask = 1
    cdef str port="C"
    cdef list res = list()
    while i<N:
        while beamC[i] == 0:
            i += 1
            if i>=N: return res
        if nonTask == 1 or i>= lastExit+timeout:
            res.append(("nonTask", port, trialNum, lastExit, i))
            lastEnd = i
            trialNum = 0
            nonTask = 0
            port= "C"
        while True:
            if beamC[i] == 1:
                lastExit = i+1
            elif beamL[i] == 1:
                if port == "R":
                    res.append(("switch", port, trialNum, lastEnd, i))
                    lastEnd = i
                    trialNum = 0
                port = "L"
                break
            elif beamR[i] == 1:
                if port == "L":
                    res.append(("switch", port, trialNum, lastEnd, i))
                    lastEnd = i
                    trialNum = 0
                port = "R"
                break
            elif i >= lastExit+timeout:
                nonTask = 1
                break
            i+= 1
            if i>=N: return res
        while i<N and (beamL[i] == 1 or beamR[i] == 1):
            i += 1
            lastExit = i
        if nonTask==0:
            trialNum += 1
            res.append(("inBlock", port, trialNum, lastEnd, lastExit))
            lastEnd = i
    return res

cdef _blockActionCoordinates(str[:] action, str[:] port, cnp.int_t[:] trialNo,
                             cnp.int_t[:] start, cnp.int_t[:] stop, Py_ssize_t N):
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[cnp.float_t, ndim=2] coordinates = np.nan*np.ones((N,2))
    cdef cnp.float_t[:] geomCdf = scipy.stats.geom.cdf(np.arange(1,100,1), 0.05)
    cdef cnp.float_t coord_x, coord_y, phi, pi=np.pi
    for i in range(port.shape[0]):
        if trialNo[i] > 1:
            coord_x = 0.5 + 5*0.5*(geomCdf[trialNo[i]-2] + geomCdf[trialNo[i]-1])
        else:
            coord_x = 0.5 + 5*0.5*geomCdf[trialNo[i]-1]
        coord_y = -1
        if port[i]=="L":
            coord_y *= -1
            coord_x *= -1
        for j in range(start[i], stop[i]):
            if action[i] == "inBlock":
                coordinates[j,0] = coord_x
                coordinates[j,1] = (0.5 + (j-start[i])/(0.0+stop[i]-start[i]))*coord_y
            elif j<start[i]+20 and action[i] == "nonTask":
                coordinates[j,0] = coord_x
                coordinates[j,1] = (1.5 + (j-start[i])/20.0)*coord_y
            elif action[i] == "switch":
                phi = pi/2*(j-start[i])/(0.0+stop[i]-start[i])
                if port[i] == "L":
                    coordinates[j,0] = (coord_x+0.5     - 1)*libc.math.sin(phi)+0.5
                    coordinates[j,1] = ((coord_x+0.5)*0.2 - 1)*libc.math.cos(phi)+0.5
                else:
                    coordinates[j,0] = (coord_x-0.5     + 1)*libc.math.cos(phi)-0.5
                    coordinates[j,1] = ((coord_x-0.5)*0.2 + 1)*libc.math.sin(phi)-0.5
    return coordinates

def blockActionCoordinates(blockActions, numFrames):
    '''
    Converts the list of block action into coordinates for plotting on top
    of the block-level task schematic.
    
    Arguments:
    blockActions -- A pandas Dataframe with the start and stop of every action, as calculated
                    by fancyViz.findBlockActions()
    numFrames -- Number of frames in recording
    
    Returns:
    A pandas Dataframe with the x and y coordinates in the block-level schematic for
    the `numFrames` frames
    '''
    return pd.DataFrame(_blockActionCoordinates(blockActions.action.values, blockActions.port.values,
                                                blockActions.trialNo.values, blockActions.start.values,
                                                blockActions.stop.values, numFrames),
                        columns=["x", "y"])

def taskSchematicCoordinatesFrameLabels(labelPerFrame):
    '''
    Similar to taskSchematicCoordinates but for the labels given by labelFrameActions instead
    of calcActionsPerFrame
    
    Returns:
    A pandas Dataframe with the x and y coordinates in the schematic for each frame
    '''
    return pd.DataFrame(_taskSchematicCoordinatesFrameLabels(labelPerFrame.label.astype("str").values,
                                                             labelPerFrame.actionProgress.values),
                        columns=["x","y"], index=labelPerFrame.index)

cdef cnp.ndarray[cnp.float_t, ndim=2] _taskSchematicCoordinatesFrameLabels(str[:] label,
                                                                cnp.float_t[:] actionProgress):
    cdef Py_ssize_t i, N = label.shape[0]
    cdef cnp.float_t x, y, normal_x, normal_y, normal_len, progress
    cdef cnp.ndarray[cnp.float_t, ndim=2] coordinates = np.nan*np.ones((N,2)) 
    cdef str port
    for i in range(N):
        progress = actionProgress[i]
        if label[i][0] == "p":
            port = label[i][1]
            if port == "L": x = -4
            elif port == "C": x = 0
            elif port == "R": x = 4
            
            if port != "C":
                if label[i][4]=='r': x *= 0.92
                else: x *= 1.08
            y = 1.0 - 2.0*progress
            if port == "C": y = -y
            coordinates[i,0] = x
            coordinates[i,1] = y
            
        elif label[i][0] == "m":
            if label[i][1] == "C":
                port = label[i][3]
                if port == "C": continue
                x = progress * 4.0
                if port == "L": x = -x
                y = 2.0*progress - 1.0
                y = 2 - y*y
                coordinates[i,0] = x
                coordinates[i,1] = y
            
            elif label[i][3] == "C":
                port = label[i][1]
                if port == "C": continue
                x = 4.0 - progress * 4.0
                if port == "L": x = -x
                y = 2.0*progress - 1.0
                y = -2 + y*y
                normal_x = 2.0*(2.0*progress - 1.0)
                normal_y = 4.0
                normal_len = libc.math.sqrt(normal_x*normal_x + normal_y*normal_y)
                if port == "L": normal_x = -normal_x
                if label[i][4] != 'r':
                    normal_x *= -1
                    normal_y *= -1
                x += normal_x / normal_len / 3.0
                y += normal_y / normal_len / 3.0
                coordinates[i,0] = x
                coordinates[i,1] = y
            
    return coordinates