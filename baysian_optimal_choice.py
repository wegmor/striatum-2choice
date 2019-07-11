import pandas as pd
import numpy as np

def optimialStrategy(chosenPort, reward, p_switch = .05, p_reward = .75):
    '''Assuming known switching and reward probabilities, compare a sequence of choices to 
    what would have been the baysian optimal sequence.
    
    Arguments:
    chosenPort -- Array of chosen ports at each trial
    reward -- Array of reward recieved (0 for no reward, >0 for reward) at each trial
    p_switch -- Probability of block change (default 5%)
    p_reward -- Probability of reward given correctlt chosen port (default 75%)
    
    Returns:
    A Pandas dataframe with the probabilities of rewarded port being to the left (p_left) and to the
    right (p_right). p_left and p_right are posterior probabilities, i.e. after checking reward. The column
    optimalChoice is instead based on the prior probabilities, i.e. before checking reward (thus a fair comparison
    to the mouse)
    '''
    T = len(chosenPort)
    p_left = np.zeros(T+1)
    p_right = np.zeros(T+1)
    optimalChoice = [""]*T
    
    #Uniform priors before first trial
    p_left[0] = 0.5
    p_right[0] = 0.5
    for i in range(1,T+1):
        j = i - 1
        
        #The prior probabilities for the rewarded port being left and right respectively
        prior_left  = (1-p_switch) * p_left[i-1] +   p_switch   * p_right[i-1]
        prior_right =   p_switch   * p_left[i-1] + (1-p_switch) * p_right[i-1]
        
        #Optimal choice according to the priors (the same information as the mouse has)
        optimalChoice[j] = "L" if prior_left > prior_right else "R"
        
        #Reward
        if reward[j] >= 1:
            if chosenPort[j] == 'L':
                p_left[i] = 1
                p_right[i] = 0
            elif chosenPort[j] == 'R':
                p_left[i] = 0
                p_right[i] = 1
        #No reward
        else:
            likelihood_left = prior_left
            likelihood_right = prior_right
            if chosenPort[j] == 'L':
                likelihood_left *= (1-p_reward)
            elif chosenPort[j] == 'R':
                likelihood_right *= (1-p_reward)
            norm = likelihood_left + likelihood_right
            p_left[i] = likelihood_left / norm
            p_right[i] = likelihood_right / norm
            
    return pd.DataFrame({'chosenPort': chosenPort, 'reward': reward, 'p_left': p_left[1:],
                         'p_right': p_right[1:], 'optimalChoice': optimalChoice})

def probabilities(chosenPort, reward, p_switch = .05, p_reward = .75):
    '''Assuming known switching and reward probabilities, given a sequence of choices find the
    probability that the reward is on either side.
    
    Arguments:
    chosenPort -- Array of chosen ports at each trial
    reward -- Array of reward recieved (0 for no reward, >0 for reward) at each trial
    p_switch -- Probability of block change (default 5%)
    p_reward -- Probability of reward given correctlt chosen port (default 75%)
    
    Returns:
    A Pandas dataframe with the probabilities of rewarded port being to the left (p_left) and to the
    right (p_right). p_left and p_right are prior probabilities, i.e. before checking reward (thus a fair
    comparison to the mouse's situation)
    '''
    T = len(chosenPort)
    p_left = np.zeros(T)
    p_right = np.zeros(T)
    optimalChoice = [""]*T
    
    #Uniform priors before first trial
    posterior_left = 0.5
    posterior_right = 0.5
    for i in range(T):
        
        #The prior probabilities for the rewarded port being left and right respectively
        p_left[i]  = (1-p_switch) * posterior_left +   p_switch   * posterior_right
        p_right[i] =   p_switch   * posterior_left + (1-p_switch) * posterior_right
        
        #Reward
        if reward[i] >= 1:
            if chosenPort[i] == 'L':
                posterior_left = 1
                posterior_right = 0
            elif chosenPort[i] == 'R':
                posterior_left = 0
                posterior_right = 1
                
        #No reward
        else:
            likelihood_left = p_left[i]
            likelihood_right = p_right[i]
            if chosenPort[i] == 'L':
                likelihood_left *= (1-p_reward)
            elif chosenPort[i] == 'R':
                likelihood_right *= (1-p_reward)
            norm = likelihood_left + likelihood_right
            posterior_left = likelihood_left / norm
            posterior_right = likelihood_right / norm
            
    return pd.DataFrame({'chosenPort': chosenPort, 'reward': reward, 'p_left': p_left,
                         'p_right': p_right})

def valueAndUncertainty(lfa, p_switch=0.05, p_reward=0.75*0.5):
    '''Calculate the external ("objective") probability of each frame being in a left block, given
    the choice history up to this point. This would reflect something akin to "value" under an
    optimal strategy. Also calculates entropy, i.e. the uncertainty, of the choice:
    E = -p*log_2(p) - (1-p)*log_2(1-p) 
    
    Arguments:
    lfa - The result from session.labelFrameActions()
    p_switch - The (objective) probability of a block switch following a reward.
    p_reward - The (objective) probability of getting a reward assuming the correct side
               is chosen. Default is 0.75 * 0.5 to reflect that about half of the trials
               are not correctly initialized.
               
    Returns:
    A pandas dataframe with the probability of each frame being in a left block (given the
    choice history), as well as the entropy of that probability.
    '''
    actions = lfa.groupby("actionNo").first()
    outcomes = actions[actions.label.str.match("p[LR].*[ro]")]
    chosenPort = outcomes.label.str[1]
    reward = outcomes.label.str[-1] == 'r'
    prob = probabilities(chosenPort.values, reward.values, p_switch, p_reward)
    prob.index = outcomes.index - 1
    prob["baysian_p_left"] = prob.p_left
    prob["baysian_uncertainty"] = -prob.p_left*np.log2(prob.p_left) - (1-prob.p_left)*np.log2(1-prob.p_left)
    prob = prob.reindex(lfa.actionNo, method="backfill").fillna(method="ffill")
    prob.index = lfa.index
    return prob[["baysian_p_left", "baysian_uncertainty"]]
