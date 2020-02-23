import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """
    
    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        self._name = "First_Choice_Hill_Climbing"
        self._theta = theta
        self.sigma = sigma
        self.evaluate = evaluationFunction
        self.numEpisodes = numEpisodes
        self.J = self.evaluate(self._theta, self.numEpisodes)

        self.init_theta = self._theta

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta

    def train(self)->np.ndarray:
        theta = np.random.normal(self._theta, self.sigma)
        J = self.evaluate(theta, self.numEpisodes)
        if J > self.J:
            self.J = J
            print("move: ", np.sum(np.abs(self._theta - theta)))
            self._theta = theta
        return self._theta

    def performance(self):
        return self.J

    def reset(self)->None:
        self._theta = self.init_theta
        self.J = self.evaluate(self._theta, self.numEpisodes)
