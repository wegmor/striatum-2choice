import numpy as np
import pandas as pd
import tqdm

#from IPython.core.debugger import set_trace

def particleFilter(tracking, nParticles = 2000, flattening=0.0, tqdmKwargs={}):
    '''
    Particle filter for calculating speeds. Currently uses the body and tailBase markers, but
    I intend to add the ear markers to this as well.
    
    See https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf for some theoretical
    motivation.
    
    Arguments:
    tracking - The tracking returned by readTracking. Some parameters are currently hardcoded
               to work well in cm, for prefer readTrackning(inCm=True).
               
    nParticles - The number of particles in the filter. More particles give better accuracy
                 but this should saturate fairly quickly.
    flattening - Set to a very small number (<<1e-10) to make the prior more flat. Helps
                 handling larger inaccuracies in the tracking at the expense of some accuracy.
                 
    Returns:
    A pandas dataframe with the estimated positions and speeds for each frame.
    '''
    
    meanElong = np.mean(np.sqrt(((tracking.body - tracking.tailBase)**2).sum(axis=1)))                    
    state = {
        'x'     : tracking.body.x.dropna().iloc[0] + np.random.normal(0,5,nParticles),
        'y'     : tracking.body.y.dropna().iloc[0] + np.random.normal(0,5,nParticles),
        'speed' : np.random.normal(0,1,nParticles),
        'bodyAngle' :      np.random.uniform(-np.pi, np.pi, nParticles),
        'bodyAngleSpeed' : np.random.normal(0, 0.2, nParticles),
         #'headAngle' :      np.random.uniform(-np.pi/2, np.pi/2, nParticles),
        'elongation':  np.abs(np.random.normal(meanElong, meanElong/2, nParticles))
    }
    means = {k: [] for k in state.keys()}
    for i in tqdm.trange(len(tracking), **tqdmKwargs):
        _step(**state)

        L = _likelihood(tracking.iloc[i], **state)

        W = np.exp(L) + flattening
        #if np.sum(W) < 1e-12:
        #    set_trace()
        W /= np.sum(W)
        for k in ("x", "y", "speed", "bodyAngleSpeed", "elongation"):
            means[k].append(np.dot(state[k], W))

        for k in ("bodyAngle",):# "headAngle"):
            xx = np.dot(W, np.cos(state[k]))
            yy = np.dot(W, np.sin(state[k]))
            means[k].append(np.arctan2(yy, xx))

        resampledInd = _resample(W)
        for k in state.keys():
            state[k] = state[k][resampledInd]
    return pd.DataFrame(means, index=tracking.index.copy())

def _likelihood(position, x, y, speed, bodyAngle, bodyAngleSpeed, elongation): #headAngle
    '''
    Calculate the log likelihoods for all particles.
    '''
    #Body
    L  = position.body.likelihood * -((x - position.body.x)**2) / 1.0
    L += position.body.likelihood * -((y - position.body.y)**2) / 1.0
    
    #Tailbase
    tailBaseX = x - elongation*np.cos(bodyAngle)
    tailBaseY = y - elongation*np.sin(bodyAngle)
    L += position.tailBase.likelihood * -((position.tailBase.x - tailBaseX)**2) / 1.0
    L += position.tailBase.likelihood * -((position.tailBase.y - tailBaseY)**2) / 1.0
    
    #Elongation
    L += -((elongation - 3.0)**2) / 1.0
    
    return L

def _step(x, y, speed, bodyAngle, bodyAngleSpeed, elongation): #headAngle
    x += speed * np.cos(bodyAngle) + np.random.normal(0, 1.0, len(x))
    y += speed * np.sin(bodyAngle) + np.random.normal(0, 1.0, len(y))
    speed += np.random.normal(0, 1.0, len(speed))
    bodyAngle += np.random.normal(bodyAngleSpeed, 0.2, len(bodyAngle)) + 2*np.pi
    #bodyAngle %= 2*np.pi
    #bodyAngle -= 2*np.pi
    bodyAngleSpeed *= 0.8
    bodyAngleSpeed += np.random.normal(0, 0.2, len(bodyAngleSpeed))
    #headAngle += np.random.normal(0, 0.5, len(bodyAngle)) + 2*np.pi
    elongation += np.random.normal(0, 0.5, len(elongation))
    #headAngle %= 2*np.pi
    #headAngle -= np.pi

def _resample(W):
    # "Systematic resample" from page 13 of Johansen (2011, see link above).
    # A little bit of numpy trickery.
    U = np.random.uniform(0,1.0/len(W)) + np.arange(len(W))/len(W)
    cumW = np.cumsum(W)
    return np.searchsorted(cumW, U)

def _dfParticles(state, sl=slice(None,None)):
    '''Convert the set or particles to a dataframe. Useful for debugging.'''
    return pd.DataFrame(state).iloc[sl]