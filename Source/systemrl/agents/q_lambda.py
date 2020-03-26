import numpy as np
from collections import defaultdict
from .td import TD


class QLambda(TD):
    def __init__(self, num_actions, gamma, lr, lmbda):
        self.name = "Q-Lambda"
        self.num_actions = num_actions
        self.q_table = defaultdict(lambda: [0.0] * self.num_actions)
        self.e = defaultdict(lambda: [0.0] * self.num_actions)
        self.gamma = gamma
        self.lr = lr
        self.lmbda = lmbda
        self.gamma_lambda = self.gamma * self.lmbda

    def name(self):
        return self.name

    def get_action(self, state):
        qsa = self.q_table[state]
        indices = np.where(qsa == np.max(qsa))[0]
        return np.random.choice(indices)

    def update(self, delta):
        for s in self.e:
            for a in range(self.num_actions):
                self.e[s][a] = self.e[s][a] * self.gamma_lambda
                self.q_table[s][a] = self.q_table[s][a] + self.lr * self.e[s][a] * delta

    def train(self, state, action, reward, next_state):
        next_action = self.get_action(next_state)
        delta = reward + self.gamma*self.q_table[next_state][next_action] - self.q_table[state][action]
        self.e[state][action] = 1.0/self.gamma_lambda 
        self.update(delta)

    def reset(self):
        self.e = defaultdict(lambda: [0.0] * self.num_actions)
