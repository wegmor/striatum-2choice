cimport numpy as cnp
from libc.math cimport sqrt, cos, sin, isnan

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import PIL
import scipy.ndimage

cdef void _incrPnt(cnp.float_t x, cnp.float_t y, cnp.float_t[:,:] canvas, cnp.float_t value):
    cdef Py_ssize_t r,c
    r = <Py_ssize_t>((y+2.5)*50)
    c = <Py_ssize_t>((x+5)*50)
    canvas[r,c] += value
    
cdef tuple _taskDensity(object[:] action,
                            object[:] port,
                            cnp.int_t[:] frameNo,
                            cnp.int_t[:] actionStart,
                            cnp.int_t[:] actionStop,
                            cnp.float_t[:] rewarded,
                            cnp.float_t[:] trace
                           ):
    cdef cnp.ndarray[cnp.float_t, ndim=2] kde = np.zeros((251,501))
    cdef cnp.ndarray[cnp.float_t, ndim=2] tracker = np.zeros((251,501))
    cdef Py_ssize_t i, N
    cdef cnp.float_t duration, progress, x, y, normal_x, normal_y
    N = frameNo.shape[0]
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
            #x += duration / 80.0
            #x += 1*((<cnp.float_t>frameNo[i] / N) - 0.5)
            y = 1.0 - 2.0*progress
            if port[i] == "C": y = -y
            _incrPnt(x,y,kde,trace[i])
            _incrPnt(x,y,tracker,1)
        elif action[i] == "centerToSide":
            x = progress * 4.0
            if port[i] == "L": x = -x
            y = 2.0*progress - 1.0
            y = 2 - y*y
            _incrPnt(x,y,kde,trace[i])
            _incrPnt(x,y,tracker,1)
        elif action[i] == "sideToCenter":
            x = 4.0 - progress * 4.0
            if port[i] == "L": x = -x
            y = 2.0*progress - 1.0
            y = -2 + y*y
            normal_x = 2.0*(2.0*progress - 1.0)
            normal_y = 4.0
            normal_len = sqrt(normal_x*normal_x + normal_y*normal_y)
            if port[i] == "L": normal_x = -normal_x
            if rewarded[i] <= 0:
                normal_x *= -1
                normal_y *= -1
            x += normal_x / normal_len / 3.0
            y += normal_y / normal_len / 3.0
            _incrPnt(x,y,kde,trace[i])
            _incrPnt(x,y,tracker,1)
    return kde, tracker

def taskDensity(apf, trace):
    return _taskDensity(apf.action.values, apf.port.values, apf.frameNo.values,
                            apf.actionStart.values, apf.actionStop.values, apf.rewarded.values, trace.values)
    
cdef tuple _trackingDensity(cnp.float_t[:] x,
                                    cnp.float_t[:] y,
                                    cnp.float_t[:] likeli,
                                    cnp.float_t[:] trace
                                   ):
    cdef cnp.ndarray[cnp.float_t, ndim=2] kde = np.zeros((304,400))
    cdef cnp.ndarray[cnp.float_t, ndim=2] tracker = np.zeros((304,400))
    cdef Py_ssize_t i, N
    N = min(x.shape[0], trace.shape[0])
    for i in range(N):
        if likeli[i] > 0.1:
            kde[<Py_ssize_t>y[i],<Py_ssize_t>x[i]] += trace[i]
            tracker[<Py_ssize_t>y[i],<Py_ssize_t>x[i]] += 1
    return kde, tracker

def trackingDensity(tracking, trace):
    head_x = 0.5*(tracking.leftEar.x + tracking.rightEar.x)
    head_y = 0.5*(tracking.leftEar.y + tracking.rightEar.y)
    likeli = tracking[[("leftEar", "likelihood"), ("rightEar", "likelihood")]].min(axis=1)
    return _trackingDensity(head_x.values, head_y.values, likeli.values, trace.values)

cdef cnp.float_t sign(cnp.float_t x):
    return (0 < x) - (x < 0)
    
cdef tuple _headDirDensity(cnp.float_t[:] headDir,
                                    cnp.float_t[:] likeli,
                                    cnp.float_t[:] trace
                                   ):
    cdef cnp.ndarray[cnp.float_t, ndim=2] kde = np.zeros((301,151))
    cdef cnp.ndarray[cnp.float_t, ndim=2] tracker = np.zeros((301, 151))
    cdef Py_ssize_t i, N, r, c
    cdef cnp.float_t turnSpeed, radius
    N = min(headDir.shape[0], trace.shape[0])
    for i in range(1,N):
        if likeli[i] > 0.1:
            turnSpeed = headDir[i] - headDir[i-1]
            
            radius = 1 + sign(headDir[i])*turnSpeed*0.5
            if radius < 0: radius = 0
            elif radius > 1.5: radius = 1.5
            c = <Py_ssize_t>(cos(headDir[i])*100*radius)
            r = <Py_ssize_t>(sin(headDir[i])*100*radius + 150)
            kde[r, c] += trace[i]
            tracker[r, c] += 1
    return kde, tracker

def headDirDensity(headDir, likeli, trace):
    return _headDirDensity(headDir.values, likeli.values, trace.values)

cdef tuple _bodyDirDensity(cnp.float_t[:] bodyDir,
                            cnp.float_t[:] projectedSpeed,
                            cnp.float_t[:] likeli,
                            cnp.float_t[:] trace
                           ):
    cdef cnp.ndarray[cnp.float_t, ndim=2] kde = np.zeros((301,301))
    cdef cnp.ndarray[cnp.float_t, ndim=2] tracker = np.zeros((301, 301))
    cdef Py_ssize_t i, N, r, c
    cdef cnp.float_t radius
    N = min(bodyDir.shape[0], trace.shape[0])
    for i in range(1,N):
        if likeli[i] > 0.1:
            radius = 1 + projectedSpeed[i] / 50.0
            if radius < 0.5:
                radius = 0.5
            elif radius > 1.5:
                radius = 1.5
            c = <Py_ssize_t>(cos(bodyDir[i])*100*radius + 150)
            r = <Py_ssize_t>(sin(bodyDir[i])*100*radius + 150)
            kde[r, c] += trace[i]
            tracker[r, c] += 1
    return kde, tracker

def bodyDirDensity(bodyDir, projectedSpeed, likeli, trace):
    return _bodyDirDensity(bodyDir.values, projectedSpeed.values, likeli.values, trace.values)

cdef tuple _bodyTurnDensity(cnp.float_t[:] bodyTurn,
                                    cnp.float_t[:] likeli,
                                    cnp.float_t[:] trace
                                   ):
    cdef cnp.ndarray[cnp.float_t, ndim=2] kde = np.zeros((301,301))
    cdef cnp.ndarray[cnp.float_t, ndim=2] tracker = np.zeros((301, 301))
    cdef Py_ssize_t i, N, r, c
    cdef cnp.float_t radius=1
    N = min(bodyTurn.shape[0], trace.shape[0])
    for i in range(1,N):
        if likeli[i] > 0.1:
            c = <Py_ssize_t>(cos(bodyTurn[i])*100*radius + 150)
            r = <Py_ssize_t>(sin(bodyTurn[i])*100*radius + 150)
            kde[r, c] += trace[i]
            tracker[r, c] += 1
    return kde, tracker

def bodyTurnDensity(bodyTurn, likeli, trace):
    return _bodyTurnDensity(bodyTurn.values, likeli.values, trace.values)

#plt.axis("off")
    
def arcArrow(rad, start, stop):
    phi = np.linspace(start,stop,100)
    plt.plot(np.cos(phi)*rad, np.sin(phi)*rad, 'k', lw=0.5)
    if start<stop:
        plt.arrow(np.cos(phi[-1])*rad, np.sin(phi[-1])*rad, -np.sin(phi[-1])*0.1, np.cos(phi[-1])*0.1,
              head_width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    else:
        plt.arrow(np.cos(phi[-1])*rad, np.sin(phi[-1])*rad, np.sin(phi[-1])*0.1, -np.cos(phi[-1])*0.1,
              head_width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
        
def drawHeadDirBackground():
    phi = np.linspace(-0.5*np.pi,0.5*np.pi,100)
    plt.plot(np.cos(phi), np.sin(phi), 'k')
    
    offs = np.pi*2 / 20 * 0.5
    arcArrow(1 + offs, -0.1, -1.4)
    arcArrow(1 + offs, 0.1, 1.4)
    arcArrow(1 - offs, -1.4, -0.25)
    arcArrow(1 - offs, 1.4,  0.25)
    
    #1
    plt.plot([0,1],[0,0], 'k--')
    plt.axis("equal")
    plt.axis("off")

def drawTaskBackground():
    plt.xlim(-5,5)
    plt.ylim(-2.75,2.5)
    elLeft = matplotlib.patches.Ellipse((-4,0),1.5,2.0,edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    elCenter = matplotlib.patches.Ellipse((0,0),1.5,2.0,edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    elRight = matplotlib.patches.Ellipse((4,0),1.5,2.0,edgecolor="k", facecolor="none", lw=2, zorder=-10000)
    plt.gca().add_artist(elLeft)
    plt.gca().add_artist(elCenter)
    plt.gca().add_artist(elRight)
    xx = np.linspace(-1,1)
    yy = 1-xx*xx
    plt.plot(xx*2+2,yy+1,'k', lw=2, zorder=-10000)
    plt.plot(xx*-2-2,yy+1,'k', lw=2, zorder=-10000)
    plt.plot(xx*2+2,-yy-1,'k', lw=2, zorder=-10000)
    plt.plot(xx*-2-2,-yy-1,'k', lw=2, zorder=-10000)
    plt.arrow(3.9,1.1,0.1,-0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.arrow(-3.9,1.1,-0.1,-0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.arrow(-0.1,-1.1,0.1,0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.arrow(0.1,-1.1,-0.1,0.1, width=0.075, length_includes_head=True, edgecolor="none", facecolor="k")
    plt.axis("off")
    
def plotTaskDensity(apf, trace, sat=0.5, smoothing=5):
    de, tracker = taskDensity(apf.reset_index(), trace)
    tracker = scipy.ndimage.gaussian_filter(tracker, smoothing)
    de = scipy.ndimage.gaussian_filter(de, smoothing)
    de /= tracker * sat
    de = np.clip(de,-1,1)

    tracker /= tracker.max()*0.02
    #absDe = np.abs(de).astype(np.float)
    colors = plt.cm.RdYlBu_r(de*0.5 + 0.5)
    colors[:,:,3] = np.clip(tracker, 0, 1)

    drawTaskBackground()
    plt.imshow(colors, vmin=-sat, vmax=sat, cmap="RdYlBu_r", origin="lower",
               interpolation="nearest", extent=(-5,5,-2.5,2.5), zorder=-100000)
    plt.axis("off")

def plotTrackingDensity(tracking, trace, sat=0.5, smoothing=3,
                        background=None):
    de, tracker = trackingDensity(tracking, trace)
    tracker = scipy.ndimage.gaussian_filter(tracker, smoothing)
    de = scipy.ndimage.gaussian_filter(de, smoothing)
    de /= tracker * sat
    de = np.clip(de,-1,1)

    tracker /= tracker.max()*0.00002
    #absDe = np.abs(de).astype(np.float)
    colors = plt.cm.RdYlBu_r(de*0.5 + 0.5)
    colors[:,:,3] = np.clip(tracker, 0, 1)
    if isinstance(background, str):
        background = PIL.Image.open(background)
    elif background is None:
        background = PIL.Image.open("/home/emil/2choice/boxBackground.png")
    
    plt.imshow(background, alpha=0.5)
    plt.imshow(colors, vmin=-sat, vmax=sat, cmap="RdYlBu_r", interpolation="nearest")
    plt.axis("off")

def calcHeadDirections(tracking):
    head = 0.5*(tracking.leftEar + tracking.rightEar)
    bodyVec = head - tracking.tailBase
    headVec = tracking.leftEar - tracking.rightEar
    norm = np.sqrt((headVec[['x','y']]**2).sum(axis=1)*(bodyVec[['x','y']]**2).sum(axis=1))
    return -np.arcsin((headVec * bodyVec)[['x','y']].sum(axis=1) / norm)    

def plotHeadDirectionDensity(headDirections, likelihood, trace, sat=0.5, smoothing=6):
    de, tracker = headDirDensity(headDirections, likelihood, trace)
    tracker = scipy.ndimage.gaussian_filter(tracker, smoothing)
    de = scipy.ndimage.gaussian_filter(de, smoothing)
    de /= tracker * sat
    de = np.clip(de,-1,1)

    tracker /= tracker.max()*0.01
    #absDe = np.abs(de).astype(np.float)
    colors = plt.cm.RdYlBu_r(de*0.5 + 0.5)
    colors[:,:,3] = np.clip(tracker, 0, 1)
    drawHeadDirBackground()
    plt.imshow(colors, vmin=-sat, vmax=sat, cmap="RdYlBu_r", interpolation="nearest", extent=(0,1.5,-1.5,1.5))
    plt.xlim(0,1.5)
    
def calcBodyDirections(tracking):
    head = 0.5*(tracking.leftEar + tracking.rightEar)
    bodyVec = head - tracking.tailBase
    return np.arctan2(bodyVec.y, bodyVec.x)

def calcProjectedSpeed(tracking):
    head = 0.5*(tracking.leftEar + tracking.rightEar)
    bodyVec = head - tracking.tailBase
    velVec = tracking.tailBase - tracking.tailBase.shift(1)
    proj = (velVec.x*bodyVec.x + velVec.y*bodyVec.y) / np.sqrt((bodyVec.x**2 + bodyVec.y**2))
    return proj

def plotBodyDirectionDensity(bodyDirections, projectedSpeed, likelihood, trace, sat=0.5, smoothing=6):
    de, tracker = bodyDirDensity(bodyDirections, projectedSpeed, likelihood, trace)
    tracker = scipy.ndimage.gaussian_filter(tracker, smoothing)
    de = scipy.ndimage.gaussian_filter(de, smoothing)
    de /= tracker * sat
    de = np.clip(de,-1,1)

    tracker /= tracker.max()*0.01
    #absDe = np.abs(de).astype(np.float)
    colors = plt.cm.RdYlBu_r(de*0.5 + 0.5)
    colors[:,:,3] = np.clip(tracker, 0, 1)
    
    plt.imshow(colors, vmin=-sat, vmax=sat, cmap="RdYlBu_r", interpolation="nearest", extent=(-1.5,1.5,-1.5,1.5))
    plt.axis("off")
    plt.axis("equal")

def plotBodyTurnDensity(turningSpeed, likelihood, trace, sat=0.5, smoothing=6):
    de, tracker = bodyTurnDensity(turningSpeed, likelihood, trace)
    tracker = scipy.ndimage.gaussian_filter(tracker, smoothing)
    de = scipy.ndimage.gaussian_filter(de, smoothing)
    de /= tracker * sat
    de = np.clip(de,-1,1)

    tracker /= tracker.max()*0.01
    #absDe = np.abs(de).astype(np.float)
    colors = plt.cm.RdYlBu_r(de*0.5 + 0.5)
    colors[:,:,3] = np.clip(tracker, 0, 1)
    
    arcArrow(1, -0.1, -2)
    arcArrow(1 , 0.1, 2)
    plt.imshow(colors, vmin=-sat, vmax=sat, cmap="RdYlBu_r", interpolation="nearest", extent=(-1.5,1.5,-1.5,1.5))
    plt.axis("off")
    plt.axis("equal")

cdef cnp.ndarray _calcGazePoint(cnp.float_t[:,:] leftEar, cnp.float_t[:,:] rightEar):
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
        head_vec_len = sqrt(headVec_x*headVec_x + headVec_y*headVec_y)
        normal_x =  headVec_y / head_vec_len
        normal_y = -headVec_x / head_vec_len
        alpha_top = (50 - head_y) / normal_y
        alpha_bottom = (250 - head_y) / normal_y
        alpha_left = (75 - head_x) / normal_x
        alpha_right = (290 - head_x) / normal_x
        inter_top = head_x + alpha_top*normal_x
        inter_bottom = head_x + alpha_bottom*normal_x
        inter_left = head_y + alpha_left*normal_y
        inter_right = head_y + alpha_right*normal_y
        if alpha_top >= 0 and inter_top > 75 and inter_top < 290:
            res[i, 0] = inter_top
            res[i, 1] = 50
        elif alpha_bottom >= 0 and inter_bottom > 75 and inter_bottom < 290:
            res[i, 0] = inter_bottom
            res[i, 1] = 250
        elif alpha_left >= 0 and inter_left > 50 and inter_left < 250:
            res[i, 0] = 75
            res[i, 1] = inter_left
        elif alpha_right >= 0 and inter_right > 50 and inter_right < 250:
            res[i, 0] = 290
            res[i, 1] = inter_right
        else:
            res[i, 0] = np.nan
            res[i, 1] = np.nan
    return res

def calcGazePoint(tracking):
    return _calcGazePoint(tracking["leftEar"][["x","y"]].values,
                          tracking["rightEar"][["x","y"]].values)

cdef tuple _gazePoint(cnp.float_t[:,:] intersectionPoints,
                      cnp.float_t[:] trace
                      ):
    cdef cnp.ndarray[cnp.float_t, ndim=2] kde = np.zeros((304,400))
    cdef cnp.ndarray[cnp.float_t, ndim=2] tracker = np.zeros((304, 400))
    cdef Py_ssize_t i, N, r, c
    cdef cnp.float_t radius=1, npNan=np.nan 
    N = min(intersectionPoints.shape[0], trace.shape[0])
    for i in range(1,N):
        if isnan(intersectionPoints[i,0]) or isnan(intersectionPoints[i,1]):
            continue
        c = <Py_ssize_t>intersectionPoints[i,0]
        r = <Py_ssize_t>intersectionPoints[i,1]
        kde[r, c] += trace[i]
        tracker[r, c] += 1
    return kde, tracker

def plotGazePoint(gazePoints, trace, sat=0.5, smoothing=3):
    de, tracker = _gazePoint(gazePoints, trace)
    tracker = scipy.ndimage.gaussian_filter(tracker, smoothing)
    de = scipy.ndimage.gaussian_filter(de, smoothing)
    de /= tracker * sat
    de = np.clip(de,-1,1)

    tracker /= tracker.max()*0.00002
    #absDe = np.abs(de).astype(np.float)
    colors = plt.cm.RdYlBu_r(de*0.5 + 0.5)
    colors[:,:,3] = np.clip(tracker, 0, 1)
    plt.imshow(colors, vmin=-sat, vmax=sat, cmap="RdYlBu_r", interpolation="nearest")
    plt.axis("off")
