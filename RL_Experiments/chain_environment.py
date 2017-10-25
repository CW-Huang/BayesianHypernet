import numpy as np


class TabularMDP(Environment):
    '''
    Tabular MDP

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    '''

    def __init__(self, nState, nAction, epLen):
        '''
        Initialize a tabular episodic MDP

        Args:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length

        Returns:
            Environment object
        '''

        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        for state in range(nState):
            for action in range(nAction):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(nState) / nState


    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = 0

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action

        Returns:
        reward - double - reward
        newState - int - new state
        pContinue - 0/1 - flag for end of the episode
        '''
        if self.R[self.state, action][1] < 1e-9:
            # Hack for no noise
            reward = self.R[self.state, action][0]
        else:
            reward = np.random.normal(loc=self.R[self.state, action][0],
                                      scale=self.R[self.state, action][1])
        newState = np.random.choice(self.nState, p=self.P[self.state, action])

        # Update the environment
        self.state = newState
        self.timestep += 1

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1

        return reward, newState, pContinue

    def compute_qVals(self):
        '''
        Compute the Q values for the environment

        Args:
            NULL - works on the TabularMDP

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    qVals[s, j][a] = self.R[s, a][0] + np.dot(self.P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])
        return qVals, qMax




def make_bootDQNChain(nState=6, epLen=15, nAction=2):
    '''
    Creates the chain from Bootstrapped DQN

    Returns:
        bootDQNChain - Tabular MDP environment
    '''
    R_true = {}
    P_true = {}

    for s in xrange(nState):
        for a in xrange(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0.01, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in xrange(nState):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(nState - 1, s + 1)] = 0.5
        P_true[s, 1][max(0, s-1)] = 0.5

    bootDQNChain = TabularMDP(nState, nAction, epLen)
    bootDQNChain.R = R_true
    bootDQNChain.P = P_true
    bootDQNChain.reset()

    return bootDQNChain
