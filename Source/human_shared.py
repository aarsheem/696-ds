import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.advgridworld import AdvGridworld
from systemrl.agents.q_learning import QLearning
import os
import random

def grid_human(coagent= None):
    """
    Gridworld with human agent that can only go up, down, left, or right and acts randomly
    Can also have an rl agent included to help guide it 90% of the time
    """
    path = os.getcwd()
    grid = AdvGridworld()
    os.chdir(path)

    if coagent is not None:
        coagenthelp = True
        print(grid.name + ' with Human and CoAgent Help')
    else:
        coagenthelp = False
        print(grid.name + ' with Human Agent Alone')

    liRewards = []
    episodes = 1000
    for i in range(episodes):
        grid.reset()
        ct =0
        while not grid.isEnd:
            #'up','right','down','left','upri','dori','dole','uple'
            actions = [0, 1, 2, 3, 4, 5, 6, 7]
            action = np.random.choice(actions, 1)[0]
            state = grid.state
            if coagenthelp:
                width = grid.getBoardDim()[0]
                state_ind = state[1]*(width+1)+state[0]

                q_vals = coagent.get_q_values(state_ind)
                best_action = coagent.get_action(state_ind)

                #random percent to decide when to help if wrong action taken
                if (best_action != action) and random.random() <= 0.9:
                    action = best_action

            grid.step(action)
            #print(grid.state)
            ct+=1
        liRewards.append(grid.reward)
    arr = np.array(liRewards)
    print("Rewards Info for ", episodes, ' Episodes')
    print('Size', arr.size)
    print('Mean', np.mean(arr, axis=0))
    print('Stdev', np.std(arr, axis=0))
    print('Min', np.min(arr))
    print('Max', np.max(arr))

    return arr

def rl_alone():
    """
    Human agent that can only go up, down, left, or right and acts randomly
    Returns the trained agent
    """
    path = os.getcwd()
    grid = AdvGridworld()
    os.chdir(path)
    print(grid.name + ' with Q-Learning Agent')
    gamma = 0.9
    learning_rate = 0.01
    agent = QLearning(8, gamma, learning_rate)

    liRewards = []
    episodes = 1000

    #action_index = {"up":0,"right":1,"down":2,"left":3,"upri":4,"dori":5,"dole":6,"uple":7}

    for i in range(episodes):
        grid.reset()
        state = grid.state
        width = grid.getBoardDim()[0]
        while not grid.isEnd:
            state_ind = state[1]*(width+1)+state[0]
            action = agent.get_action(state_ind)
            next_state, reward, is_end = grid.step(action)
            next_state_ind = next_state[1]*(width+1)+next_state[0]
            agent.train(state_ind, action, reward, next_state_ind)
            state = next_state
        #print(i, grid.reward)
        liRewards.append(grid.reward)
    arr = np.array(liRewards)
    print("Rewards Info for ", episodes, ' Episodes')
    print('Size', arr.size)
    print('Mean', np.mean(arr, axis=0))
    print('Stdev', np.std(arr, axis=0))
    print('Min', np.min(arr))
    print('Max', np.max(arr))

    return arr, agent

if __name__ == "__main__":
    grid_human()
    print()
    _, rl_agent = rl_alone()
    print()
    grid_human(coagent=rl_agent)
