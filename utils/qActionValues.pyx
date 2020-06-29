cimport numpy as cnp
import numpy as np
cdef void calcQValues(cnp.float_t[:] output, cnp.int8_t[:] rightPort,
                      cnp.int8_t[:] rewarded, cnp.float_t alpha):
    cdef Py_ssize_t i, N = output.shape[0]
    cdef cnp.float_t Q_l = 0.0, Q_r = 0.0
    output[0] = 0.0
    for i in range(N-1):
        if rightPort[i]: Q_r += alpha*(rewarded[i] - Q_r)
        else: Q_l += alpha*(rewarded[i] - Q_l)
        output[i+1] = Q_r - Q_l
        
def qActionValues(rightPort, rewarded, alpha):
    actionValues = np.empty(rightPort.shape[0])
    calcQValues(actionValues, rightPort, rewarded, alpha)
    return actionValues

def qLikelihood(rightPort, rewarded, alpha, beta, s):
    actionValues = qActionValues(rightPort, rewarded, alpha)
    L_r = -np.logaddexp(0, -beta*actionValues-s)
    L_l = -np.logaddexp(0,  beta*actionValues+s)
    L = np.where(rightPort, L_r, L_l)
    return np.sum(L)

