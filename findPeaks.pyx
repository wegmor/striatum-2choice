cimport numpy as cnp
import numpy as np
import pandas as pd

cpdef list findPeaksDonahue(cnp.float_t[:,:] caTraces):
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

def findPeaks(caTraces, method="donahue"):
    if method != "donahue":
        raise NotImplementedError(method + " not implemented")
    peaks = pd.DataFrame(findPeaksDonahue(caTraces.values), columns=["frameNo", "neuron"])
    return peaks.join(pd.Series(caTraces.index), on="frameNo")