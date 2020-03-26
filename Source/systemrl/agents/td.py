from abc import ABC, abstractmethod
from .skeleton import Agent


class TD(Agent):
    """
    An Agent that employs TD learning techniques.
    """ 

    @abstractmethod
    def get_action(self, state):
        """
        Returns the action corresponding to the maximum q value.
        """
        pass

    @abstractmethod
    def train(self, state, action, reward, next_state):
        """
        One update step of the q-table. Does not return anything.
        """
        pass
