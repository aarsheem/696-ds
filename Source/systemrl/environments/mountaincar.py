import gym
import numpy as np
from typing import Tuple
from .skeleton import Environment


class Mountaincar(Environment):
    def __init__(self):
        self._name = "MountainCar-v0"
        self.env = gym.make(self._name).env
        self._state = self.env.reset()
        self._action = None
        self._reward = 0
        self._isEnd = False

        self.xMax = -1.2
        self.xMin = 0.5
        self.vMax = -0.07
        self.vMin = 0.07

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> int:
        return self._state

    def step(self, action: int):
        self._action = action
        self._state, self._reward, self._isEnd, _ = self.env.step(action)
        return self._state, self._reward, self._isEnd

    def reset(self) -> None:
        self._state = self.env.reset()
        self._action = None
        self._reward = 0
        self._isEnd = False

    def normState(self):
        """
        Normalize state values in range 0 -- 1
        """
        x = (self._state[0] - self.xMin)/(self.xMax - self.xMin)
        v = (self._state[1] - self.vMin)/(self.vMax - self.vMin)
        return np.array([x,v])

    def R(self, state: int, action: int, nextState: int) -> float:
        return 1

    def numFeatures(self):
        return 2

    def numActions(self):
        return 3

    def gamma(self):
        return 1

    def nextState(self, state, action):
        pass
