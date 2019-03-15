import numpy as np
import pandas as pd
import sklearn.svm
import sklearn.model_selection

def pairwiseStateDecoding(sess, groupA, groupB, includeRewards=False, includeSwitches=False, nFolds=5, scoring="balanced_accuracy"):
    '''
    Use a linear support vector machine to decode from the deconvolved calcium traces which
    of two task states (as defined by labelFrameActions) the animal is in. 
    
    Arguments:
    sess - The session. Will be used to find det deconvolved signal and the labelFrameActions.
    groupA - A list of strings containing the state labels to include in the first group. Use
             a list with a single element to only include one state.
    groupB - The list of states to compare to groupA, on the same format.
    includeRewards  - Passed on to sess.labelFrameActions. Make sure this matches the group specifications.
    includeSwitches - Passed on to sess.labelFrameActions. Make sure this matches the group specifications.
    nFolds - The trials will be split into this number of folds. For each fold, an SVM will be trained on
             the remaining `nFolds-1` folds and then tested on the fold.
    scoring - How to measure score. Typically use 'accuracy' or 'balanced_accuracy', but see also
              https://scikit-learn.org/stable/modules/model_evaluation.html

    Returns:
    An array with nFold elements, the test score (according to scoring) of each fold when the SVM is
    trained on the other folds.
    '''
    deconv = sess.readDeconvolvedTraces()
    deconv /= deconv.std()
    lfa = sess.labelFrameActions(reward=includeRewards, switch=includeSwitches)
    if len(lfa) != len(deconv):
        raise ValueError("Sensor values and calcium signal have different lengths for session {}.".format(sess))
    avgSig = deconv.groupby(lfa.actionNo.values).mean()
    labels = lfa.groupby("actionNo").label.first()
    labels[labels.isin(groupA)] = 0
    labels[labels.isin(groupB)] = 1
    okTrials = labels.isin([0,1])
    okTrials[avgSig.isna().any(axis=1).values] = False
    X = avgSig[okTrials].values
    Y = labels[okTrials].values.astype(np.bool)
    classifier = sklearn.svm.LinearSVC()
    splitter = sklearn.model_selection.StratifiedKFold(nFolds, shuffle=True)
    scores = sklearn.model_selection.cross_val_score(classifier, X, Y, cv=splitter, scoring=scoring, n_jobs=-1)
    return scores


def findStates(trials, maxDur = 30):
    '''
    An alternative to sess.labelFrameActions that are instead based on sess.findTrials.
    
    Arguments:
    trials - Trials returned by sess.findTrials.
    maxDur - The maximum number of frames allowed for a movement between ports (does not apply
             to the time spent inside ports).
             
    Returns:
    A DataFrame with the start and stop frames of each state.
    '''
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

def decodingWithDifferentNumberOfNeurons(sess, nFolds=5, stepSize=10):
    '''
    Try decoding the full state space using an increasing number of neurons. State space
    is defined by findStates in this module (and not labelFrameActions).
    
    Arguments:
    sess - The session object. This method does not handle missing frames.
    nFolds - The number of cross-validation folds. The set of selected neurons
             is also changed between each folds.
    stepSize - The number of neurons included are increased in steps of this size.
    
    Returns:
    A DataFrame with the number of correctly predicted states per number of neurons.
    '''
    
    trials = sess.findTrials()
    states = findStates(trials)

    deconv = sess.readDeconvolvedTraces()
    deconv /= (deconv.std()+1e-10)
    if deconv.isna().any().any():
        raise ValueError("This function does not support dropped frames.")
    avgSig = []
    for p in states.itertuples():
        avgSig.append(deconv.iloc[p.start:p.stop].mean())
    avgSig = np.array(avgSig)

    for nNeurons in tqdm.trange(stepSize, deconv.shape[1], stepSize, desc=str(sess)):
        for i in range(nFolds):
            splitter = sklearn.model_selection.StratifiedKFold(5, shuffle=True)
            nStates = len(states.state.cat.categories)
            selectedNeurons = np.random.choice(np.arange(deconv.shape[1]), nNeurons, False)
            signal = avgSig[:, selectedNeurons]
            train, test = next( splitter.split(signal, states.state) )
            classifier = sklearn.svm.LinearSVC()
            classifier.fit(signal[train,:], states.state.iloc[train])
            pred = classifier.predict(signal[test,:])
            correct = np.sum(pred==states.state.iloc[test])
            total = len(pred)
            res.append((str(sess), nNeurons, i, correct, total))
    return pd.DataFrame(res, columns=["session", "nNeurons", "repetition", "correct", "total"])