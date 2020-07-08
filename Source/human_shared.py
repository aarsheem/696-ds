import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.advgridworld_old import AdvGridworld
from systemrl.agents.q_learning import QLearning
import seaborn as sns
import os
import random

"""
This file is for the implementation of the baseline plots for the paper.
"""

def grid_human_co(episodes = 1000, h_agent=None, coagent= None, threshold = 1.0, verbose = True, mode='override'):
    """
    Gridworld with human agent and co-pilot if desired
    Default is uniform random actions

    Coded to only work with old version of Advanced Gridworld
    i.e. gridworld1 with no breaking but with diagonal movements

    Parameters
    -------------------------------------
    episodes: number of episodes to run
    h_agent: a specified human agent (not used and left to default random human)
    coagent: the rl agent meant to cooperate with the human
    threshold: the constraint value 0 to 1 (alpha in Shared Autonomy)
    verbose: printing options
    mode: how do we want to cooperate; q-diff (Shared Autonomy), q-neg, or override
          if left blank, the human agent will play alone

    Returns
    -------------------------------------
    reward: list of evniornment rewards for each episode
    rewards_penalized: list of intervention rewards for each episode
    actionsctli: list of intervention count for each episode
    """

    path = os.getcwd()
    grid = AdvGridworld()
    os.chdir(path)

    #this is just printing stuff for easier terminal reading
    if h_agent is not None:
        hstr = "Specialized Human"
    else:
        hstr = "Random Human"
    if coagent is not None:
        coagenthelp = True
        if verbose: print(grid.name + ' with '+hstr+' and CoAgent('+ mode +')')
    else:
        coagenthelp = False
        if verbose: print(grid.name + ' with '+hstr+' Agent Alone')

    liRewards = []
    liRewardsPenalized = []
    actionsctli =[]
    heatmap = np.zeros((7, 7))
    for i in range(episodes):
        grid.reset()
        actionct = 0
        misstepct = 0
        heatmap[0,0] += 1
        while (not grid.isEnd) and (grid.numSteps <=200):
            state = grid.state
            width = grid.getBoardDim()[0]
            state_ind = state[1]*(width+1)+state[0]

            if 'disjoint' in mode:
                actions = [0, 1, 2, 3]
            else:
                #'up','right','down','left','upri','dori','dole','uple'
                actions = [0, 1, 2, 3, 4, 5, 6, 7]
            #if we have a special human agent use it, if not use random human
            if h_agent is None:
                action = np.random.choice(actions, 1)[0]
                h_action = action
            else:
                action = h_agent.get_action(state_ind)
                h_action = action
            if coagenthelp:
                q_vals = coagent.get_q_values(state_ind)
                best_action = coagent.get_action(state_ind)
                best_q = q_vals[best_action]
                min_q = min(q_vals)
                human_q = q_vals[action]

                #closest actions dictionary
                closest_a = {0:[4,7],1:[4,5],2:[5,6],3:[6,7],4:[0,1],5:[1,2],6:[2,3],7:[0,3]}
                #value = action index if radial
                #key = actual action index
                a_dist_ind = {0:0, 1:2, 2:4, 3:6, 4:1, 5:3 ,6:5 ,7:7}
                #actions in order of circle
                a_radial = [0,4,1,5,2,6,3,7]
                #co-pilot action choices
                #override to optimal if suboptimal action selected
                if "override" in mode:
                    #random percent to decide when to help if wrong action taken
                    if (human_q < best_q) and np.random.uniform() <= 1-threshold:
                        actionct +=  1
                        action = best_action
                #select best closest action if human action q-value is negative
                elif mode == "q-neg" and human_q < 0:
                    actionct += 1
                    a_li = closest_a[action]
                    action = a_li[np.argmax([q_vals[a] for a in a_li])]
                #Shared Autonomy
                #select better closest action if difference between best q and human q is past some threshold
                elif mode == "q-diff" and (human_q-min_q)<(1-threshold)*(best_q-min_q):
                    actionct += 1
                    for i in range(1,5):
                        radial_a = a_dist_ind[action]
                        ccw_ra = (radial_a - i) % len(actions)
                        ccw_a = a_radial[ccw_ra]
                        if q_vals[ccw_a]-min_q >= (1-threshold)*(best_q-min_q):
                            action = ccw_a
                            break
                        cw_ra = (radial_a + i) % len(actions)
                        cw_a = a_radial[cw_ra]
                        if q_vals[cw_a]-min_q >= (1-threshold)*(best_q-min_q):
                            action = cw_a
                            break
            #take the potentiallny new action
            grid.step(action)

            if h_agent is not None:
                heatmap[grid.state[1],grid.state[0]] += 1

        #add metrics to their respective lists
        liRewards.append(grid.reward)
        liRewardsPenalized.append(grid.reward-actionct)
        actionsctli.append(actionct)
    #convert lists to numpy arrays
    rewards = np.array(liRewards)
    rewards_penalized = np.array(liRewardsPenalized)
    actionsctli = np.array(actionsctli)
    #print some metrics if we want to
    if verbose:
        print("Rewards Info for ", episodes, ' Episodes')
        print('Size', rewards.size)
        print('Mean', np.mean(rewards, axis=0))
        print('Stdev', np.std(rewards, axis=0))
        print('Min', np.min(rewards))
        print('Max', np.max(rewards))
        print('Avg num of interventions:', np.mean(actionsctli, axis=0))
        #heatmap makes a nice drawing for the paths taken if we have a specialized human
        if h_agent is not None:
            plt.figure()
            print(heatmap)
            plt.imshow(heatmap, cmap='cool', interpolation='nearest')
            plt.savefig('two.png')

    return rewards, actionsctli, rewards_penalized

def rl_alone(episodes = 1000, fill_table = False):
    """
    rl agent trained with q-learning for gridworld
    can use fill-table to create a q-table by randomly taking actions
    Returns the trained agent

    Parameters
    -------------------------------------
    episodes: number of episodes to run
    fill_table: sets whether we want to fill the Q-table aswell

    Returns
    -------------------------------------
    reward: list of rewards for each episode
    agent: the trained agent
    """
    path = os.getcwd()
    grid = AdvGridworld()
    os.chdir(path)
    print(grid.name + ' with Q-Learning Agent')
    gamma = 0.95
    learning_rate = 0.01
    agent = QLearning(8, gamma, learning_rate)

    liRewards = []

    #action_index = {"up":0,"right":1,"down":2,"left":3,"upri":4,"dori":5,"dole":6,"uple":7}

    for i in range(episodes):
        grid.reset()
        state = grid.state
        width = grid.getBoardDim()[0]
        while not grid.isEnd:
            state_ind = state[1]*(width+1)+state[0]
            #If we want a filled Q-tables, we can't let the agent be optimal
            #or else we wouldn't know the q-values for never reached state-action pairs
            if fill_table:
                actions = [0, 1, 2, 3, 4, 5, 6, 7]
                action = np.random.choice(actions, 1)[0]
            else:
                action = agent.get_action(state_ind)
            next_state, reward, is_end = grid.step(action)
            next_state_ind = next_state[1]*(width+1)+next_state[0]
            agent.train(state_ind, action, reward, next_state_ind)
            state = next_state

        liRewards.append(grid.reward)
    rewards = np.array(liRewards)
    print("Rewards Info for ", episodes, ' Episodes')
    print('Size', rewards.size)
    print('Mean', np.mean(rewards, axis=0))
    print('Stdev', np.std(rewards, axis=0))
    print('Min', np.min(rewards))
    print('Max', np.max(rewards))

    return rewards, agent

def q_human(episodes = 1000, fill_table = False):
    """
    This was mostly unused for the paper but will be left here.

    human agent trained with q-learning for gridworld
    for now try to make a #2 shape
    can use fill-table to create a q-table by randomly taking actions
    Returns the trained agent
    """
    path = os.getcwd()
    grid = AdvGridworld()
    os.chdir(path)
    print(grid.name + ' with Q-Learning Human Agent')
    gamma = 0.95
    learning_rate = 0.01
    agent = QLearning(8, gamma, learning_rate)

    liRewards = []

    for i in range(episodes):
        grid.reset()
        state = grid.state
        width = grid.getBoardDim()[0]
        top_right_unex = True
        bottom_left_unex = True
        while not grid.isEnd:
            state_ind = state[1]*(width+1)+state[0]
            if fill_table:
                actions = [0, 1, 2, 3, 4, 5, 6, 7]
                action = np.random.choice(actions, 1)[0]
            else:
                action = agent.get_action(state_ind)
            next_state, reward, is_end = grid.step(action)
            next_state_ind = next_state[1]*(width+1)+next_state[0]
            #makes a 2 shape
            if (next_state_ind not in [0,1,2,3,10,17,24,23,22,21,28,35,42,43,44,45,46,47,48]):
                reward = -10

            agent.train(state_ind, action, reward, next_state_ind)
            state = next_state
        #print(i, grid.reward)
        liRewards.append(grid.reward)
    rewards = np.array(liRewards)
    print("Rewards Info for ", episodes, ' Episodes')
    print('Size', rewards.size)
    print('Mean', np.mean(rewards, axis=0))
    print('Stdev', np.std(rewards, axis=0))
    print('Min', np.min(rewards))
    print('Max', np.max(rewards))

    return rewards, agent

def main_paper(mode = 'q-diff'):
    """
    Main method for generating plots for paper

    Parameters
    -------------------------------------
    mode: the mode used for human-rl cooperation
    """
    mode_name = 'Shared Autonomy'
    if mode== 'override':
        mode_name = 'Overrider'

    #trains rl again and fills a q-table
    _, rl_agent = rl_alone(1000, fill_table = True)

    #prints the q-table
    for y in range(7):
        for x in range(7):
            state_ind = y*(7)+x
            q = rl_agent.get_q_values(state_ind)
            print('Q-table')
            print(state_ind, [x,y], np.around(q, decimals=5))

    in_li = []
    r_li = []
    rp_li=[]

    print("Begin looping through constraint values")
    for i in np.arange(0.0, 1.025, 0.025):
        #cooperates and gets results
        rewards, actionsct, rewardsP = grid_human_co(coagent=rl_agent, threshold=i, verbose=False, mode=mode)
        print("Threshold: ", i)
        avgInter = np.mean(actionsct, axis=0)
        avgR = np.mean(rewards, axis=0)
        avgRP = np.mean(rewardsP, axis=0)
        in_li.append(avgInter)
        r_li.append(avgR)
        rp_li.append(avgRP)
        print('Avg num of interventions: ', avgInter)
        print('Avg reward: ', avgR)
        print('Avg reward penalized: ', avgRP)
        print()

    plt.figure()
    plt.plot(np.arange(0.0, 1.025, 0.025), in_li)
    plt.title(mode_name + ' Interventions')
    plt.xlabel(r"$\alpha$")
    plt.ylabel('Average intervention count')
    plt.show()

    plt.figure()
    plt.title(mode_name + ' Returns', fontsize=14)
    plt.xlabel(r"$\alpha$", fontsize=13)
    plt.ylabel('Average return', fontsize=13)
    plt.plot(np.arange(0.0, 1.025, 0.025), r_li,label='Average environment return')
    plt.plot(np.arange(0.0, 1.025, 0.025), rp_li,label="Average intervention return")
    plt.legend(fontsize=9)
    plt.show()

if __name__ == "__main__":
    main_paper(mode='q-diff')
