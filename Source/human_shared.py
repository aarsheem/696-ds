import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.advgridworld import AdvGridworld
from systemrl.agents.q_learning import QLearning
import os
import random

def grid_human_co(episodes = 1000, h_agent=None, coagent= None, threshold = 1.0, verbose = True, mode='override'):
    """
    Gridworld with human agent and co-pilot if desired
    Default is uniform random actions
    """
    path = os.getcwd()
    grid = AdvGridworld(4)
    os.chdir(path)

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
    actionsctli =[]
    heatmap = np.zeros((7, 7))
    for i in range(episodes):
        grid.reset()
        ct =0
        actionct = 0
        misstepct = 0
        while (not grid.isEnd):# and (grid.numSteps <=200):
            state = grid.state
            width = grid.getBoardDim()[0]
            state_ind = state[0][1]*(width)+state[0][0]
            init_state = (state_ind, state[1:])

            #'up','right','down','left','upri','dori','dole','uple'
            actions = [0, 1, 2, 3, 4, 5]
            #if we have a special human agent, use it
            if h_agent is None:
                action = np.random.choice(actions, 1)[0]
            else:
                action = h_agent.get_action(state_ind)
            #print(action)
            if coagenthelp:
                q_vals = coagent.get_q_values(state_ind)
                #print(q_vals)
                best_action = coagent.get_action(state_ind)
                best_q = q_vals[best_action]
                human_q = q_vals[action]

                #closest actions dictionary
                #closest_a = {0:[4,7],1:[4,5],2:[5,6],3:[6,7],4:[0,1],5:[1,2],6:[2,3],7:[0,3]}
                closest_a = {0: [1, 3], 1: [0, 2], 2: [3, 1], 3: [2, 0], 4: [5, 1], 5: [4, 1]}

                #co-pilot action choices
                #override to optimal if suboptimal action selected
                if mode == "override":
                    #random percent to decide when to help if wrong action taken
                    if (human_q < best_q) and np.random.uniform() <= threshold:
                        actionct +=  1
                        action = best_action
                #select best closest action if human action q-value is negative
                elif mode == "q-neg" and human_q < 0:
                    actionct += 1
                    a_li = closest_a[action]
                    action = a_li[np.argmax([q_vals[a] for a in a_li])]
                #select better closest action if difference between best q and human q is past some threshold
                elif mode == "q-diff" and human_q<(1-threshold)*best_q:
                    actionct += 1
                    a_li = closest_a[action]
                    prev_a_li = [action]
                    #if the immediate closest actions are not better check the next pair of closest actions
                    #needs to be optimzed as it takes a while to run
                    while q_vals[action] <= human_q:
                        action = a_li[np.argmax([q_vals[a] for a in a_li])]
                        if q_vals[action] <= human_q:
                            new_a_li = []
                            for a in a_li:
                                new_a = closest_a[a]
                                for a1 in new_a:
                                    if a1 not in prev_a_li:
                                        new_a_li.append(a1)
                            a_li = new_a_li.copy()
                            prev_a_li.extend(a_li)
            """print(action)
            print(grid.state)
            prev_state = grid.state
            prev_action = action"""
            grid.step(action)
            #print(grid.state)
            if h_agent is not None:
                heatmap[grid.state[1],grid.state[0]] += 1
                new_state = grid.state
                new_state_ind = new_state[0][1]*(width)+new_state[0][0]
                new_state_full = (new_state_ind, state[1:])
                #print(new_state)
                if (new_state_ind not in [0,1,2,3,10,17,24,23,22,21,28,35,42,43,44,45,46,47,48]):
                    misstepct +=1
            ct+=1
        if grid.numSteps > 200:
            r = grid.reward#-50.0
        else:
            r = grid.reward-misstepct
        liRewards.append(r)
        actionsctli.append(actionct)
    rewards = np.array(liRewards)
    actionsctli = np.array(actionsctli)
    if verbose:
        print("Rewards Info for ", episodes, ' Episodes')
        print('Size', rewards.size)
        print('Mean', np.mean(rewards, axis=0))
        print('Stdev', np.std(rewards, axis=0))
        print('Min', np.min(rewards))
        print('Max', np.max(rewards))
        print('Avg num of interventions:', np.mean(actionsctli, axis=0))
        #heatmap makes a nice drawing for the paths taken
        if h_agent is not None:
            plt.figure()
            print(heatmap)
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.show()
    return rewards, actionsctli

def rl_alone(episodes = 1000, fill_table = False):
    """
    rl agent trained with q-learning for gridworld
    can use fill-table to create a q-table by randomly taking actions
    Returns the trained agent
    """
    path = os.getcwd()
    grid = AdvGridworld(3)
    os.chdir(path)
    print(grid.name + ' with Q-Learning Agent')
    gamma = 0.95
    learning_rate = 0.01
    agent = QLearning(6, gamma, learning_rate)

    liRewards = []

    #action_index = {"up":0,"right":1,"down":2,"left":3,"upri":4,"dori":5,"dole":6,"uple":7}

    for i in range(episodes):
        grid.reset()
        state = grid.state
        width = grid.getBoardDim()[0]
        while not grid.isEnd:
            state_ind = state[0][1]*(width)+state[0][0]
            #new
            init_state = (state_ind, state[1:])
            if fill_table:
                actions = [0, 1, 2, 3, 4, 5]
                action = np.random.choice(actions, 1)[0]
            else:
                action = agent.get_action(state_ind)
            next_state, reward, is_end = grid.step(action)
            next_state_ind = next_state[0][1]*(width)+next_state[0][0]
            #new
            next_state_full = (next_state_ind, state[1:])
            agent.train(state_ind, action, reward, next_state_ind)
            state = next_state
            #print(state)
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

def q_human(episodes = 1000, fill_table = False):
    """
    human agent trained with q-learning for gridworld
    for now try to make a #2 shape
    can use fill-table to create a q-table by randomly taking actions
    Returns the trained agent
    """
    path = os.getcwd()
    grid = AdvGridworld(1)
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
            #to make the agent want to reach the corners
            """if bottom_left_unex and next_state_ind == 42:
                bottom_left_unex = False
                reward = 10 * (gamma**(grid.numSteps-1))
            elif top_right_unex and next_state_ind == 6:
                top_right_unex = False
                reward = 10 * (gamma**(grid.numSteps-1))"""

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

if __name__ == "__main__":
    """_, h_agent = q_human(episodes=1000, fill_table=True)
    print()
    for y in range(7):
        for x in range(7):
            state_ind = y*(7)+x
            q = h_agent.get_q_values(state_ind)
            print(state_ind, [x,y], np.around(q, decimals=5))"""
    _, rl_agent = rl_alone(1000, fill_table = True)
    print()
    for y in range(7):
        for x in range(7):
            state_ind = y*(7)+x
            q = rl_agent.get_q_values(state_ind)
            print(state_ind, [x,y], np.around(q, decimals=5))

    """in_li = []
    r_li = []
    #takes a while because of better closest action search loops
    for i in np.arange(0.1, 1.025, 0.025):
        rewards, actionsct = grid_human_co(h_agent=h_agent,coagent=rl_agent, threshold = i, verbose = False)
        print("Threshold: ", i)
        avgInter = np.mean(actionsct, axis=0)
        avgR = np.mean(rewards, axis=0)
        in_li.append(avgInter)
        r_li.append(avgR)
        print('Avg num of interventions: ', avgInter)
        print('Avg reward: ', avgR)

        print()
    plt.figure()
    plt.plot(np.arange(0.1, 1.025, 0.025), in_li)
    plt.title('Override Suboptimal with Best Interventions')
    plt.xlabel('Probabilty to override')
    plt.ylabel('Average intervention count')
    plt.savefig("overideinterventionstwo.png")

    plt.figure()
    plt.plot(np.arange(0.1, 1.025, 0.025), r_li)
    plt.title('Override Suboptimal with Best Rewards')
    plt.xlabel('Probabilty to override')
    plt.ylabel('Average reward')
    plt.savefig("overiderewardstwo.png")"""

    print()
    in_li = []
    r_li = []
    for i in np.arange(0.0, 1.01, 0.01):
        print("Threshold: ", i)
        rewards, actionsct = grid_human_co(episodes=1000,coagent=rl_agent, threshold=i, verbose=False, mode='q-diff')

        avgInter = np.mean(actionsct, axis=0)
        avgR = np.mean(rewards, axis=0)
        in_li.append(avgInter)
        r_li.append(avgR)
        print('Avg num of interventions: ', avgInter)
        print('Avg reward: ', avgR)

        print()
    plt.figure()
    plt.plot(np.arange(0.0, 1.01, 0.01), in_li)
    plt.title('Q-Value Difference with Closest Action Interventions')
    plt.xlabel('Multi threshold')
    plt.ylabel('Average intervention count')
    plt.savefig("qmdiffinterventionstwo.png")

    plt.figure()
    plt.plot(np.arange(0.0, 1.01, 0.01), r_li)
    plt.title('Q-Value Difference with Closest Action Rewards')
    plt.xlabel('Multi threshold')
    plt.ylabel('Average reward')
    plt.savefig("qmdiffrewardstwo.png")

    #grid_human_co(h_agent=h_agent,coagent=rl_agent,mode='q-neg')
    print()
