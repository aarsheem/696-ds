import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10):
        self._name = "Genetic_Algorithm"
        self.populationSize = populationSize
        self.evaluate = evaluationFunction
        self.initPopulation  = initPopulationFunction
        self.numElite = numElite
        self.numEpisodes = numEpisodes
        self.J = np.zeros(self.populationSize)
        self.alpha = 0.1
        self.T = 1
        self._population = self.initPopulation(self.populationSize) 

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._population[0]

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        return parent + self.alpha * np.random.normal(size=parent.shape)        

    def get_parents(self):
        return np.random.choice(self.T, self.populationSize-self.numElite)
    
    def performance(self):
        return np.mean(self.J[:self.numElite])

    def train(self)->np.ndarray:
        for i in range(self.populationSize):
            self.J[i] = self.evaluate(self._population[i], self.numEpisodes)
        #reverse sort
        jindex = np.argsort(self.J)[::-1]
        self._population = self._population[jindex]
        self.J = self.J[jindex]
        #select (populationSize - numElite) children
        parentIndices = self.get_parents()
        for i in range(self.numElite, self.populationSize):
            self._population[i] = self._mutate(self._population[parentIndices[i-self.numElite]])
        return self._population[0]


    def reset(self)->None:
        self._population = self.initPopulation(self.populationSize)
