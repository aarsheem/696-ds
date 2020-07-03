import gym
import pylab
import numpy as np
from advgridworld import AdvGridworld
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale

from maxent import *

n_states = 49
n_actions = 8
one_feature = 20 # number of state per one feature
q_table = np.zeros((n_states, n_actions)) # (400, 3)
print(q_table.shape)
feature_matrix = np.eye((n_states)) # (400, 400)

gamma = 0.95
q_learning_rate = 0.03
theta_learning_rate = 0.005

np.random.seed(1)

def idx_demo(env, one_feature):
    # env_low = env.observation_space.low
    # env_high = env.observation_space.high
    # env_distance = (env_high - env_low) / one_feature

    raw_demo = np.load(file="expert_demo/expert_demo.npy", allow_pickle = True)
    print(raw_demo.shape)
    # demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))

    # for x in range(len(raw_demo)):
    #     for y in range(len(raw_demo[0])):
    #         position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
    #         velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
    #         state_idx = position_idx + velocity_idx * one_feature

    #         demonstrations[x][y][0] = state_idx
    #         demonstrations[x][y][1] = raw_demo[x][y][2]

    return raw_demo

def idx_state(env, state):
    # env_low = env.observation_space.low
    # env_high = env.observation_space.high
    # env_distance = (env_high - env_low) / one_feature
    # position_idx = int((state[0] - env_low[0]) / env_distance[0])
    # velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    # state_idx = position_idx + velocity_idx * one_feature
    return state

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    #print(state)
    #print(q_2)
    q_table[state][action] += q_learning_rate * (q_2 - q_1)


def main():
    env = AdvGridworld()

    demonstrations = idx_demo(env, one_feature)

    expert = expert_feature_expectations(feature_matrix, demonstrations)
    print(expert.reshape((7,7)))
    plt.figure()
    plt.title("Trajectory Feature Expectation")
    sns.heatmap(expert.reshape(7,7))
    plt.show()
    learner_feature_expectations = np.zeros(n_states)

    theta = -(np.random.uniform(size=(n_states,)))

    episodes, scores = [], []
    heatmap = np.zeros((7, 7))
    for episode in range(10000):
        env.reset()
        state = [0,0]
        score = 0


        if (episode != 0 and episode == 50) or (episode > 50 and episode % 50 == 0):
            learner = learner_feature_expectations / episode
            maxent_irl(expert, learner, theta, theta_learning_rate)

        while True:

            state_idx = state[1]*(env.getBoardDim()[0]+1)+state[0]
            action = np.argmax(q_table[state_idx])
            if np.random.uniform() > 0.95:
                next_state, reward, done = env.step(np.random.randint(0,7))
            else:
                next_state, reward, done = env.step(action)
            #print(q_table)
            #print("action: ", action, reward)

            irl_reward = get_reward(feature_matrix, theta, n_states, state_idx)
            next_state_idx = next_state[1]*(env.getBoardDim()[0]+1)+next_state[0]
            update_q_table(state_idx, action, irl_reward, next_state_idx)

            learner_feature_expectations += feature_matrix[int(state_idx)]

            score += reward
            state = next_state
            if episode >1000:
                heatmap[state[1],state[0]]+=1

            if done:
                if episode % 1000 == 0:
                    print(reward)
                scores.append(score)
                episodes.append(episode)
                break

        if episode % 1000 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./learning_curves/maxent_30000.png")
            np.save("./results/maxent_q_table", arr=q_table)
            print(q_table)
    plt.figure()
    plt.title("MaxEnt Feature Expectation")
    sns.heatmap((learner_feature_expectations/10000).reshape(7,7))
    plt.show()

    print(q_table)
    plt.figure()

    sns.heatmap(heatmap)
    plt.show()
if __name__ == '__main__':
    main()
