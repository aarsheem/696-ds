import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import os


ENV_NAME = "Breakout-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def breakout():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]* env.observation_space.shape[1]* env.observation_space.shape[2]
    action_space = env.action_space.n
    dqn_solver_left = DQNSolver(observation_space, 2)
    dqn_solver_right = DQNSolver(observation_space, 2)
    run = 0
    cur_path = os.getcwd()
    while True:
        print("Run : ",run)
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        total_score = 0
        while True:
            dqn_solver_left_action = dqn_solver_left.act(state)
            dqn_solver_right_action = dqn_solver_right.act(state)
            

            if dqn_solver_left_action == 1 and dqn_solver_right_action == 1:
                action = 1
            elif dqn_solver_left_action == 1 and dqn_solver_right_action == 0:
                action = 3
            elif  dqn_solver_left_action == 0 and dqn_solver_right_action == 1:
                action = 2
            else:
                action = 1

            #env.render()
            
            state_next, reward, terminal, info = env.step(action)
            total_score += reward
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver_left.remember(state, dqn_solver_left_action, reward, state_next, terminal)
            dqn_solver_right.remember(state, dqn_solver_right_action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("LEFT Run: " + str(run) + ", exploration: " + str(dqn_solver_left.exploration_rate) + ", score: " + str(total_score))
                print("RIGHT Run: " + str(run) + ", exploration: " + str(dqn_solver_right.exploration_rate) + ", score: " + str(total_score))
                file_path = os.path.join(cur_path,'breakout_'+str(run)+'.h5')
                dqn_solver_left.model.save(file_path)
                break
            dqn_solver_left.experience_replay()
            dqn_solver_right.experience_replay()


    


if __name__ == "__main__":
    breakout()