import numpy as np
from collections import defaultdict
from .td import TD


class SARSA(TD):
    def __init__(self, num_actions, gamma, lr):
        self.name = "SARSA"
        self.num_actions = num_actions
        self.q_table = defaultdict(lambda: [0.0] * self.num_actions)
        self.gamma = gamma
        self.lr = lr
        self.state = None

    def name(self):
        return self.name

    def get_action(self, state):
        qsa = self.q_table[state]
        indices = np.where(qsa == np.max(qsa))[0]
        return np.random.choice(indices)

    def train(self, state, action, reward, next_state):
        if self.state is None:
            self.state = state
            self.action = action
            self.reward = reward
            return
        self.q_table[self.state][self.action] += self.lr*(self.reward + self.gamma\
                *self.q_table[state][action] - self.q_table[self.state][self.action])
        self.state = state
        self.action = action
        self.reward = reward

    def reset(self):
        self.q_table[self.state][self.action] += self.lr*(self.reward - self.q_table[self.state][self.action])
        self.state = None
