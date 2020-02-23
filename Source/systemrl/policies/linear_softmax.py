import numpy as np
from .skeleton import Policy
from typing import Union

class LinearSoftmax(Policy):
    """

    Parameters
    ----------
    numFeatures (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numFeatures:int, numActions:int, k=1):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self.numStates = (k+1)**numFeatures
        self.numActions = numActions
        self._theta = np.zeros((self.numStates, self.numActions))
        self.k = k
        self.C = np.zeros((self.numStates,numFeatures))
        for i in range(self.numStates):
            curr_i = i
            #order is important
            for j in range(numFeatures-1, -1, -1):
                self.C[i][j] = curr_i%(k+1)
                curr_i = int(curr_i/(k+1))
        #print("C: ")
        #print(self.C)
                
    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.T.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape, order='F')

    def __call__(self, state:np.ndarray, action=None)->Union[float, np.ndarray]:
        psa = self.getActionProbabilities(state)
        if action:
            return psa[action]
        return psa

    def sampleAction(self, state:np.ndarray)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        psa = self.getActionProbabilities(state)
        if np.isnan(psa).any():
            print("Probability of an action is Nan!")
            return np.random.randint(self.numActions)
        return np.random.choice(self.numActions, p=psa)

    def getActionProbabilities(self, state:np.ndarray)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        state = self.feature_transform(state)
        thetaDotS = np.matmul(self._theta.T, state)
        expTheta = np.exp(thetaDotS - np.max(thetaDotS))
        psa = expTheta/sum(expTheta)
        return psa

    def getActionGivenStatesProbabilities(self, states):
        """
        Compute the softmax action probabilities for multiple states
        """
        states = self.feature_transform(states)
        thetaDotS = np.matmul(self._theta.T, states)
        expTheta = np.exp(thetaDotS - np.max(thetaDotS,0))
        psa = expTheta/sum(expTheta)
        return psa.T

    def feature_transform(self, state):
        state = np.array(state).T
        return np.cos(np.pi*np.matmul(self.C, state))


if __name__ == "__main__":
    numFeatures = 5
    numActions = 3
    k = 2
    ls = LinearSoftmax(numFeatures, numActions, k)
    ls.parameters = np.random.random(729)
    s = np.random.random((2, numFeatures))
    print(ls.getActionProbabilities(s[0]))
    print(ls.getActionProbabilities(s[1]))
    print(ls.getActionGivenStatesProbabilities(s))
