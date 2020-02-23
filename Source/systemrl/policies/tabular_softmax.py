import numpy as np
from .skeleton import Policy
from typing import Union

class TabularSoftmax(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates:int, numActions: int):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self._theta = np.zeros((numStates, numActions))
        self.numStates = numStates
        self.numActions = numActions
        #p(s,a) = 1/numActions
        self.psa = np.ones((numStates, numActions))/numActions
                
    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)
        for s in range(self.numStates):
            expTheta = np.exp(self._theta[s,:] - max(self._theta[s,:]))
            self.psa[s] = expTheta/sum(expTheta)

    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        if action is not None:
            return self.psa[state][action]
        return self.psa[state]

    def sampleAction(self, state:int)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        #return np.argmax(self.psa[state,:])
        if np.isnan(self.psa).any():
            return np.random.randint(self.numActions)
        return np.random.choice(self.numActions, p=self.psa[state,:])

    def getActionProbabilities(self, state:int)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        return self.psa[state,:]

