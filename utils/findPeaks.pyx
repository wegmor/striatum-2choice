cimport numpy as cnp
import numpy as np
import pandas as pd

def findPeaks(caTraces, method="donahue", shape="long"):
    if method != "donahue":
        raise NotImplementedError(method + " not implemented")  
    if not isinstance(caTraces, pd.DataFrame):
        raise TypeError("Expected caTraces to be a Pandas DataFrame")
        
    if shape=="long":
        peaks = pd.DataFrame(findPeaksDonahueLong(caTraces.values), columns=["frameNo", "neuron"])
        return peaks.join(pd.Series(caTraces.index), on="frameNo")
    elif shape=="wide":
        return pd.DataFrame(findPeaksDonahueWide(caTraces.values), columns=caTraces.columns, index=caTraces.index)
    else:
        raise NotImplementedError("Shape " + shape + " not implemented")
        
cpdef list findPeaksDonahueLong(cnp.float_t[:,:] caTraces):
    cdef Py_ssize_t nNeurons = caTraces.shape[1]
    cdef Py_ssize_t T = caTraces.shape[0]
    cdef Py_ssize_t n, i, j
    cdef cnp.float_t[:] margins = np.std(caTraces, axis=0)
    cdef list peaks = []
    for n in range(nNeurons):
        for i in range(1,T-1):
            if caTraces[i,n] > caTraces[i-1,n] and caTraces[i,n] > caTraces[i+1,n]:
                j = i
                while j>0 and caTraces[j-1,n] < caTraces[j,n]:
                    j -= 1
                if caTraces[i,n] > caTraces[j,n] + margins[n]:
                    peaks.append((j, n))
    return peaks

cpdef cnp.ndarray[cnp.float_t, ndim=2] findPeaksDonahueWide(cnp.float_t[:,:] caTraces):
    cdef Py_ssize_t nNeurons = caTraces.shape[1]
    cdef Py_ssize_t T = caTraces.shape[0]
    cdef Py_ssize_t n, i, j
    cdef cnp.float_t[:] margins = np.std(caTraces, axis=0)
    cdef cnp.ndarray[cnp.float_t, ndim=2] peaks = np.zeros((T, nNeurons))
    for n in range(nNeurons):
        for i in range(1,T-1):
            if caTraces[i,n] > caTraces[i-1,n] and caTraces[i,n] > caTraces[i+1,n]:
                j = i
                while j>0 and caTraces[j-1,n] < caTraces[j,n]:
                    j -= 1
                if caTraces[i,n] > caTraces[j,n] + margins[n]:
                    peaks[j, n] += 1.0
    return peaks