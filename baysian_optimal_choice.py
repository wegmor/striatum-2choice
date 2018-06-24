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