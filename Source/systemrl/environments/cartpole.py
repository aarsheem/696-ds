import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"
        
        # TODO: properly define the variables below
        self._action = None
        self._reward = 0
        self._isEnd = False
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable

        self.xMin = -2.4
        self.xMax = 2.4
        self.vMin = -10
        self.vMax = 10
        self.thetaMin = -np.pi / 12.0
        self.thetaMax = np.pi / 12.0
        self.omegaMin = -np.pi
        self.omegaMax = np.pi


    @property
    def name(self)->str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array((self._x, self._v, self._theta, self._dtheta))

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        dstate = np.zeros(4)
        dstate[0] = state[1]
        
        dstate[2] = state[3]
        
        F = action * 20.0 - 10.0
        cos_multiplier = (-F - self._mp * self._l * (state[3]**2) * np.sin(state[2])) / (self._mp + self._mc)
        denominator = self._l * (4.0/3.0 - (self._mp * (np.cos(state[2])**2))/(self._mp + self._mc))
        dstate[3] = (self._g * np.sin(state[2]) + np.cos(state[2]) * cos_multiplier) / denominator
        
        ml_multipier = (dstate[2]**2) * np.sin(state[2]) - dstate[3] * np.cos(state[2]) 
        dstate[1] = (F + self._mp * self._l * ml_multipier) / (self._mp + self._mc)
        return state + dstate * self._dt

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        #note the new reward
        #at 15 degrees reward will be -1
        return np.cos(12 * state[2])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        next_state = self.nextState(self.state, action)
        self._reward = self.R(self.state, action, next_state)
        self._action = action

        self._x = next_state[0]
        self._v = next_state[1]
        self._theta = next_state[2]
        self._dtheta = next_state[3]

        self._t += self._dt
        self._isEnd = self.terminal()
        return next_state, self._reward, self._isEnd

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self._isEnd = False
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole
        self._t = 0
        self._action = None
    
    def normState(self):
        """
        Normalize state values in range 0 -- 1
        """
        x = (self._x - self.xMin)/(self.xMax - self.xMin)
        v = (self._v - self.vMin)/(self.vMax - self.vMin)
        #to spread out the distribution
        x = (x - 0.5) * 10 + 0.5
        v = (v - 0.5) * 5 + 0.5
        theta = (self._theta - self.thetaMin)/(self.thetaMax - self.thetaMin)
        dtheta = (self._dtheta - self.omegaMin)/(self.omegaMax - self.omegaMin)
        return np.array([x,v,theta,dtheta])

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        if self._t > 20:
            return True
        if np.abs(self._theta) > np.pi/12.0:
            return True
        if np.abs(self._x) >= 3.0:
            return True
        return False
   
    def numActions(self):
       return 2

    def numFeatures(self):
       return 4

