import pymc3 as pm
import numpy as np
import pandas as pd
import pyximport; pyximport.install()
from striatum_2choice import trackingGeometryUtils

class Tobit(pm.Normal):
    '''PyMC3 distribution for the Tobit model (see https://en.wikipedia.org/wiki/Tobit_model).'''
    
    def random(self, point=None, size=None):
        samples = pm.Normal.random(self, point, size)
        return samples * (samples > 0)
    
    def logp(self, value):
        lp  = (value>0)  * pm.Normal.logp(self, value)
        lp += (value<=0) * (pm.math.log(pm.math.erfc(self.mean*pm.math.sqrt(self.tau/2))) - pm.math.log(2))
        #Alternate, but (surprisingly) slightly slower option:
        #lp  = pm.math.where(value>0,
        #                    pm.Normal.logp(self, value),
        #                    pm.math.log(pm.math.erfc(self.mean*np.sqrt(self.tau/2))) - np.log(2))
        return lp

def calcTurningSpeed(tracking):
    headCoordinates = (0.5*(tracking.leftEar + tracking.rightEar))[['x','y']]
    likelihood = tracking[[("leftEar", "likelihood"),
                       ("rightEar", "likelihood"),
                       ("tailBase", "likelihood")]].min(axis=1)
    bodyDir = trackingGeometryUtils.calcBodyDirection(tracking)
    turningSpeed = np.clip(trackingGeometryUtils.angleDiff(bodyDir.shift(1), bodyDir), -np.pi/2, np.pi/2)
    turningSpeed[likelihood.values<0.9] = 0
    return pd.Series(turningSpeed).fillna(0)

def findTaskVariables(block, tiltTerm=True, bleach=True, baseline=True):
    queries = {
        'L2C': "action=='sideToCenter' & port=='L' & actionDuration <= 20",
        'R2C': "action=='sideToCenter' & port=='R' & actionDuration <= 20",
        'C2L': "action=='centerToSide' & port=='L' & actionDuration <= 20",
        'C2R': "action=='centerToSide' & port=='R' & actionDuration <= 20",

        'inL': "action=='inPort' & port=='L'",
        'inC': "action=='inPort' & port=='C'",
        'inR': "action=='inPort' & port=='R'",

        'rewL': "action=='inPort' & port=='L' & rewarded==1 & frameNo-actionStart >= 7",
        'rewR': "action=='inPort' & port=='R' & rewarded==1 & frameNo-actionStart >= 7"
    }
    
    apf = block.calcActionsPerFrame()
    variables = pd.DataFrame(index=apf.index)
    
    for k,q in queries.items():
        on = apf.eval(q)
        variables[k] = on
        if tiltTerm: variables[k+"_tilt"] = np.where(on, apf.actionProgress-0.5, 0)
            
    tracking = block.readTracking()
    variables["turningSpeed"] = calcTurningSpeed(tracking)
    speed = trackingGeometryUtils.calcProjectedSpeed(tracking)
    variables["runningSpeed"] = pd.Series(np.where(tracking.tailBase.likelihood > 0.9, speed, 0)).fillna(0)
    
    if bleach:
        variables["bleach"] = np.linspace(0,1,len(variables))
        
    if baseline:
        variables["baseline"] = 1
    
    return variables


def findStates(trials, maxDur = 30):
    res = []
    for trial in trials.itertuples():
        if trial.previousPort in "RL" and trial.enterCenter-trial.exitPrevious<=maxDur:
            s2cMid = (trial.exitPrevious + trial.enterCenter)//2
            res.append(("%s2C_1"%trial.previousPort, trial.exitPrevious, s2cMid))
            res.append(("%s2C_2"%trial.previousPort, s2cMid, trial.enterCenter))
        if trial.chosenPort in "RL" and trial.enterSide-trial.exitCenter<=maxDur:
            #res.append(("inC_h%s"%trial.chosenPort, trial.enterCenter, trial.exitCenter))
            res.append(("inC", trial.enterCenter, trial.exitCenter))
            c2sMid = (trial.enterSide + trial.exitCenter)//2
            res.append(("C2%s_1"%trial.chosenPort, trial.exitCenter, c2sMid))
            res.append(("C2%s_2"%trial.chosenPort, c2sMid, trial.enterSide))
        if trial.chosenPort in "RL":
            res.append(("in%s_w"%trial.chosenPort, trial.enterSide, trial.enterSide+7))
            if trial.enterSide+7 < trial.exitSide:
                if trial.reward:
                    res.append(("in%s_r"%trial.chosenPort, trial.enterSide+7, trial.exitSide))
                else:
                    res.append(("in%s_o"%trial.chosenPort, trial.enterSide+7, trial.exitSide))
    res = pd.DataFrame(res, columns=["state", "start", "stop"])
    res.state = pd.Categorical(res.state)
    return res.query("stop - start > 0")
    