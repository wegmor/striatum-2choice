cimport numpy as cnp
cimport libc.math

import numpy as np
import pandas as pd

def calcGazePoint(tracking, boundingRect=(75, 250, 290, 50)):
    '''
    Find the point on the wall where the mouse is tentatively looking, based
    on the position and direction of its head.
    
    Arguments:
    tracking -- The pandas Dataframe given by block.readTracking()
    boudningRect -- The coordinates of the rectangle describing the
                    walls as a tuple (left, bottom, right, top).
                    
    Returns:
    A pandas Dataframe where each row contains the x and y coordinates of the point on the
    wall for the corresponding frame. All coordinates will be on the border of boundingRect,
    or be NaN when no intersection can be found (typcially due to failed tracking).
    '''
    if len(boundingRect) != 4:
        raise ValueError("Argument boundingRect must have four elements.")
    values = _calcGazePoint(tracking["leftEar"][["x","y"]].values,
                          tracking["rightEar"][["x","y"]].values,
                          boundingRect[0], boundingRect[1], boundingRect[2], boundingRect[3])
    return pd.DataFrame(values, columns=["x", "y"], index=tracking.index.copy())

def calcProjectedSpeed(tracking):
    '''
    Calculate the speed of the mouse in the direction of its body orientation, i.e.
    $\vec{\delta} \cdot \hat{b}$, where $\vec{delta}$ is the vector difference in position of
    the tail base between this and the previous frame, and $\hat{b}$ is the normalized body
    direction vector (tail base to middle point between ears). Note that this speed can be
    negative when the mouse is backing.
    
    Arguments:
    tracking -- The pandas Dataframe given by block.readTracking()
                    
    Returns:
    A pandas Series with the projected speed 
    '''
    head = 0.5*(tracking.leftEar + tracking.rightEar)
    bodyVec = head - tracking.tailBase
    velVec = tracking.tailBase - tracking.tailBase.shift(1)
    proj = (velVec.x*bodyVec.x + velVec.y*bodyVec.y) / np.sqrt((bodyVec.x**2 + bodyVec.y**2))
    return proj.rename("projectedSpeed")

def calcBodyDirection(tracking):
    '''
    Calculate the body direction of the mouse, i.e. the allocentric angle of
    the vector from the tail base to middle point between ears.
    
    Arguments:
    tracking -- The pandas Dataframe given by block.readTracking()
                    
    Returns:
    A pandas Series with the direction in radians. Following mathematical convention, 0 radians
    points to the right of the video, positive values points to the top and negative values to the
    bottom.
    '''
    head = 0.5*(tracking.leftEar + tracking.rightEar)
    bodyVec = head - tracking.tailBase
    return np.arctan2(bodyVec.y, bodyVec.x).rename("bodyDirection")

def calcHeadDirections(tracking):
    '''
    Calculate the head direction of the mouse, i.e. the egocentric angle between the body vector
    and the head vector (the normal to the line between the right and left ears).
    
    Arguments:
    tracking -- The pandas Dataframe given by block.readTracking()
                    
    Returns:
    A pandas Series with the head direction in radians. 0 radians indicates looking straight ahead,
    positive values indicates looking to the left and negative values indicates looking to the right.
    '''
    head = 0.5*(tracking.leftEar + tracking.rightEar)
    bodyVec = head - tracking.tailBase
    headVec = tracking.leftEar - tracking.rightEar
    norm = np.sqrt((headVec[['x','y']]**2).sum(axis=1)*(bodyVec[['x','y']]**2).sum(axis=1))
    return -np.arcsin((headVec * bodyVec)[['x','y']].sum(axis=1) / norm).rename("headDirection")

def angleDiff(a, b):
    return np.arctan2(np.sin(b-a), np.cos(b-a))

cdef cnp.ndarray _calcGazePoint(cnp.float_t[:,:] leftEar, cnp.float_t[:,:] rightEar,
                               cnp.float_t leftWall_x=75.0, cnp.float_t bottomWall_y=250.0,
                               cnp.float_t rightWall_x=290.0, cnp.float_t topWall_y=50.0
                              ):
    cdef Py_ssize_t i, N=leftEar.shape[0]
    cdef cnp.float_t head_x, head_y
    cdef cnp.float_t headVec_x, headVec_y, head_vec_len
    cdef cnp.float_t normal_x, normal_y
    cdef cnp.float_t alpha_top, alpha_bottom, alpha_left, alpha_right
    cdef cnp.float_t inter_top, inter_bottom, inter_left, inter_right
    cdef cnp.ndarray[cnp.float_t, ndim=2] res = np.zeros((N, 2))
    for i in range(N):
        head_x = 0.5*(leftEar[i,0] + rightEar[i,0])
        head_y = 0.5*(leftEar[i,1] + rightEar[i,1])
        headVec_x = rightEar[i,0] - leftEar[i,0]
        headVec_y = rightEar[i,1] - leftEar[i,1]
        head_vec_len = libc.math.sqrt(headVec_x*headVec_x + headVec_y*headVec_y)
        normal_x =  headVec_y / head_vec_len
        normal_y = -headVec_x / head_vec_len
        alpha_top = (topWall_y - head_y) / normal_y
        alpha_bottom = (bottomWall_y - head_y) / normal_y
        alpha_left = (leftWall_x - head_x) / normal_x
        alpha_right = (rightWall_x - head_x) / normal_x
        inter_top = head_x + alpha_top*normal_x
        inter_bottom = head_x + alpha_bottom*normal_x
        inter_left = head_y + alpha_left*normal_y
        inter_right = head_y + alpha_right*normal_y
        if alpha_top >= 0 and inter_top > leftWall_x and inter_top < rightWall_x:
            res[i, 0] = inter_top
            res[i, 1] = topWall_y
        elif alpha_bottom >= 0 and inter_bottom > leftWall_x and inter_bottom < rightWall_x:
            res[i, 0] = inter_bottom
            res[i, 1] = bottomWall_y
        elif alpha_left >= 0 and inter_left > topWall_y and inter_left < bottomWall_y:
            res[i, 0] = leftWall_x
            res[i, 1] = inter_left
        elif alpha_right >= 0 and inter_right > topWall_y and inter_right < bottomWall_y:
            res[i, 0] = rightWall_x
            res[i, 1] = inter_right
        else:
            res[i, 0] = np.nan
            res[i, 1] = np.nan
    return res

