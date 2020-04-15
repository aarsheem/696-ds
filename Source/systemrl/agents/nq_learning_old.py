import numpy as np
from collections import defaultdict
from .td import TD


class NQLearningOld(TD):
    def __init__(self, q2star_table, lr1, lr2, s0, threshold):
        self.name = "nQ-Learning"
        self.num_actions = len(q2star_table[0])
        self.q1_table = defaultdict(lambda: [0.0] * self.num_actions)
        self.q2_table = q2star_table
        self.v2_table = {}
        for state in self.q2_table:
            self.v2_table[state] = np.max(self.q2_table[state])
        self.lr1 = lr1
        self.lr2 = lr2
        self.s0 = s0
        self.threshold = threshold
        self.flag = 0

    def name(self):
        return self.name
    
    def get_action(self, state, epsilon=0):
        if np.random.random() < epsilon:
            q2sa = self.q2_table[state]
            indices = np.where(q2sa == np.max(q2sa))[0]
            return np.random.choice(indices)
        else:
            return self.get_train_action(state)

    def get_train_action(self, state):
        q1sa = self.q1_table[state]
        nq2sa = np.array(self.q2_table[state]) + self.v2_table[self.s0]\
                - self.v2_table[state]
        flag = False
        maxq1sa = -np.inf
        maxq2sa = -np.inf
        for action in range(self.num_actions):
            if flag:
                if nq2sa[action] >= self.threshold and q1sa[action] > maxq1sa:
                    maxq1sa = q1sa[action]
                    curr_action = action
            else:
                if nq2sa[action] >= self.threshold:
                    maxq1sa = q1sa[action]
                    curr_action = action
                    flag = True
                elif nq2sa[action] > maxq2sa:
                    maxq2sa = nq2sa[action]
                    curr_action = action
        if flag:
            self.flag += 1
        return curr_action
    
    def train(self, state, action, reward1, reward2, next_state):
        next_action = self.get_train_action(next_state)
        self.q1_table[state][action] += self.lr1*(reward1 + self.q1_table\
                [next_state][next_action] - self.q1_table[state][action])
        delta_q2 = reward2 + self.q2_table[next_state][next_action] - \
                self.q2_table[state][action]
        delta_v2 = reward2 + self.v2_table[next_state] - self.v2_table[state]
        self.q2_table[state][action] += self.lr2 * delta_q2
        self.v2_table[state] += self.lr2 * delta_q2
        
    def get_flag(self):
        ans = self.flag
        self.flag = 0
        return ans


    def get_q_values(self, state):
        return self.q1_table[state], self.q2_table[state]

    def reset(self):
        pass
