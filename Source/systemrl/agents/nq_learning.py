import numpy as np
from collections import defaultdict
from .td import TD


class NQLearning(TD):
    def __init__(self, q2star_table, gamma, lr, min_performance):
        self.name = "nQ-Learning"
        self.num_actions = len(q2star_table[0])
        self.q1_table = defaultdict(lambda: [0.0] * self.num_actions)
        self.q2_table = q2star_table
        self.lr = lr
        self.min_performance = min_performance
        self.gamma = gamma
        self.G2 = 0 #past discounted returns
        self.gamma_powert = 1
        self.flag_count = 0

    def name(self):
        return self.name

    def check_action(self, state, action):
        nq2sa = self.gamma_powert * np.array(self.q2_table[state]) + self.G2
        if nq2sa[action] >= self.min_performance:
            return action
        return self.get_action(state)

    def get_action(self, state):
        q1sa = self.q1_table[state]
        nq2sa = self.gamma_powert * np.array(self.q2_table[state]) + self.G2
        flag = False
        maxq1sa = -np.inf
        maxq2sa = -np.inf
        for action in range(self.num_actions):
            if flag:
                if nq2sa[action] >= self.min_performance and q1sa[action] > maxq1sa:
                    maxq1sa = q1sa[action]
                    curr_action = action
            else:
                if nq2sa[action] >= self.min_performance:
                    maxq1sa = q1sa[action]
                    curr_action = action
                    flag = True
                elif nq2sa[action] > maxq2sa:
                    maxq2sa = nq2sa[action]
                    curr_action = action
        if flag:
            self.flag_count += 1
        return curr_action

    #call when train is not being called
    def update(self, reward, is_end):
        self.G2 += reward * self.gamma_powert
        self.gamma_powert = self.gamma_powert * self.gamma
        if is_end: 
            self.gamma_powert = 1
            self.G2 = 0
    
    def train(self, state, action, reward1, reward2, next_state, is_end):
        self.G2 += reward2 * self.gamma_powert
        self.gamma_powert = self.gamma_powert * self.gamma
        next_action = self.get_action(next_state)
        delta_q1 = reward1 + self.q1_table[next_state][next_action] - \
                self.q1_table[state][action]
        delta_q2 = reward2 + self.q2_table[next_state][next_action] - \
                self.q2_table[state][action]
        self.q1_table[state][action] += self.lr * delta_q1
        self.q2_table[state][action] += self.lr * delta_q2
        if is_end: 
            self.gamma_powert = 1
            self.G2 = 0
        
    def get_flag(self):
        ans = self.flag
        self.flag = 0
        return ans

    def get_q_values(self, state):
        return self.q1_table[state], self.q2_table[state]

    def reset(self):
        pass
