import gym
import pylab
import numpy as np
from advgridworld import AdvGridworld
import matplotlib.pyplot as plt
import seaborn as sns

q_table = np.load(file="results/maxent_q_table.npy") # (400, 3)
one_feature = 20 # number of state per one feature

def idx_to_state(env, state):
    """ Convert pos and vel about mounting car environment to the integer value"""
    # env_low = env.observation_space.low
    # env_high = env.observation_space.high
    # env_distance = (env_high - env_low) / one_feature
    # position_idx = int((state[0] - env_low[0]) / env_distance[0])
    # velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    # state_idx = position_idx + velocity_idx * one_feature
    return state

def main():

    episodes, scores = [], []
    env = AdvGridworld()
    n = 2000
    scoresli =0
    heatmap = np.zeros((7, 7))
    for episode in range(n):
        state = [0,0]
        score = 0
        env.reset()
        heatmap[state[1],state[0]]+=1

        while True:
            # env.render()
            state_idx = state[1]*(env.getBoardDim()[0]+1)+state[0]
            action = np.argmax(q_table[state_idx])
            if np.random.uniform() > 0.95:
                next_state, reward, done = env.step(np.random.randint(0,7))
            else:
                next_state, reward, done = env.step(action)

            score += reward
            state = next_state
            heatmap[state[1],state[0]]+=1
            #print(state)
            if done:
                scoresli += score
                scores.append(score)
                episodes.append(episode)
                # pylab.plot(episodes, scores, 'b')
                #plt.hist(scores)
                #plt.title("scores in " + str(n)+ " tests")
                #plt.savefig("./learning_curves/maxent_test.png")
                break


        if episode % 1 == 0:
            print('{} episode score is {:.2f}'.format(episode, score))
    print(scoresli/10000)
    plt.figure()
    plt.title("Trained Path")
    sns.heatmap(heatmap)
    plt.show()
if __name__ == '__main__':
    main()
