import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable, epsilon:float=0.0001):
        self._name = "Cross_Entropy_Method"
        self._theta = theta
        #covariance matrix
        self._Sigma = np.eye(theta.shape[0]) * sigma
        self.popSize = popSize
        self.numElite = numElite
        self.numEpisodes = numEpisodes
        self.evaluate = evaluationFunction
        self.epsilon = epsilon
        self.J = np.zeros(self.popSize)

        self.init_theta = self._theta
        self.init_Sigma = self._Sigma
        
    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #todo: not sure
        return self._theta
        #return np.append(self._theta, self._Sigma.flatten)

    def covariance(self, theta):
        diff_theta = theta - self._theta
        cov = (np.matmul(diff_theta.T, diff_theta) + self.epsilon
        * np.eye(theta.shape[1]))/(self.numElite + self.epsilon)
        return cov

    def train(self)->np.ndarray:
        theta = np.random.multivariate_normal(self._theta, self._Sigma, self.popSize)
        for k in range(self.popSize):
            self.J[k]  = self.evaluate(theta[k], self.numEpisodes)
        #reverse sort
        jindex = np.argsort(self.J)[::-1]
        theta = theta[jindex]
        J = self.J[jindex]
        self._theta = np.mean(theta[:self.numElite], 0)
        self._Sigma = self.covariance(theta[:self.numElite])
        #print("Sigma: ",np.sum(np.abs(self._Sigma)))
        return theta[0]

    def performance(self):
        return np.mean(self.J[:self.numElite])


    def reset(self)->None:
        self._theta = self.init_theta
        self._Sigma = self.init_Sigma
