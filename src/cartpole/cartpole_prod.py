import random
import gym
import gym_cust
import random
import msvcrt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import time
import os


UP_ARROW =  65362
DOWN_ARROW = 65364
LEFT_ARROW = 65361
RIGHT_ARROW = 65363

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




env = gym.make('gym_cust:cartpolenoop-v1')

DEFAULT_ACTION = 2
human_agent_action = DEFAULT_ACTION


def key_press(key, mod):
    global human_agent_action
    a = 2
    if key == RIGHT_ARROW:
        a = 0

    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a=2
    if key == RIGHT_ARROW:
        a = 0
    if human_agent_action == a:
        human_agent_action = 2

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release



def cartpole():
    print("Action Space : ",env.action_space)
    print("Observation Space : ",env.observation_space)
    observation_space = env.observation_space.shape[0]
    run = 0
    dqn_solver = DQNSolver(observation_space, 2)
    cur_path = os.getcwd()
    file_path = os.path.join(cur_path, 'models', 'cartpole','cart_model_gold_standard1.h5')
    dqn_solver.model = load_model(file_path)
    while True:
        env.reset()
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 1
        while True:
            ai_action = dqn_solver.act(state)
            a = human_agent_action
            step += 1
            if a == 0:
                user_action = 1
            else:
                user_action = 0
        
            if ai_action == 1 and user_action == 1:
                action = 2
            elif ai_action == 1 and user_action == 0:
                action = 0
            elif  ai_action == 0 and user_action == 1:
                action = 1
            else:
                action = 2

            
            env.render("human")
            state_next, reward, terminal, info = env.step(action)
            state = state_next
            if terminal:
                print("Reward : ",step)
                break
            time.sleep(0.05)

if __name__ == "__main__":
    cartpole()