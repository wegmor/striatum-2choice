%%cython
#cython: unraisable_tracebacks=True  
cimport numpy as cnp
import numpy as np

cdef void incrPnt(cnp.float_t x, cnp.float_t y, cnp.float_t[:,:] canvas, cnp.float_t value):
    cdef Py_ssize_t r,c
    r = <Py_ssize_t>((y+2.5)*50)
    c = <Py_ssize_t>((x+5)*50)
    canvas[r,c] += value
    
cdef tuple _densityEstimate(object[:] action,
                            object[:] port,
                            cnp.int_t[:] frameNo,
                            cnp.int_t[:] actionStart,
                            cnp.int_t[:] actionStop,
                            cnp.float_t[:] trace
                           ):
    cdef cnp.ndarray[cnp.float_t, ndim=2] kde = np.zeros((251,501))
    cdef cnp.ndarray[cnp.float_t, ndim=2] tracker = np.zeros((251,501))
    cdef Py_ssize_t i, N;
    cdef cnp.float_t duration, progress, x, y;
    N = frameNo.shape[0];
    for i in range(N):
        duration = actionStop[i] - actionStart[i]
        if duration==0: continue
        else: progress = (frameNo[i] - actionStart[i]) / duration
        if action[i] == "inPort":
            if port[i] == "L": x = -4
            elif port[i] == "C": x = 0
            elif port[i] == "R": x = 4
            #x += duration / 80.0
            #x += 1*((<cnp.float_t>frameNo[i] / N) - 0.5)
            y = 1.0 - 2.0*progress
            if port[i] == "C": y = -y
            incrPnt(x,y,kde,trace[i])
            incrPnt(x,y,tracker,1)
        elif action[i] == "centerToSide":
            x = progress * 4.0
            if port[i] == "L": x = -x
            y = 2.0*progress - 1.0
            y = 2 - y*y
            incrPnt(x,y,kde,trace[i])
            incrPnt(x,y,tracker,1)
        elif action[i] == "sideToCenter":
            x = 4.0 - progress * 4.0
            if port[i] == "L": x = -x
            y = 2.0*progress - 1.0
            y = -2 + y*y
            incrPnt(x,y,kde,trace[i])
            incrPnt(x,y,tracker,1)
    return kde, tracker

def densityEstimate(apf, trace):
    return _densityEstimate(apf.action.values, apf.port.values, apf.frameNo.values,
                            apf.actionStart.values, apf.actionStop.values, trace.values)