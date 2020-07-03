import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.advgridworld import AdvGridworld
from systemrl.agents.q_learning import QLearning
from inverserl.gridworld.maxent.maxent import *
import seaborn as sns
import os
import random

def grid_human_co(episodes = 1000, h_agent=None, coagent= None, threshold = 1.0, verbose = True, mode='override', iota = None):
    """
    Gridworld with human agent and co-pilot if desired
    Default is uniform random actions
    """
    path = os.getcwd()
    grid = AdvGridworld()
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
    liRewardsPenalized = []
    actionsctli =[]
    heatmap = np.zeros((7, 7))
    for i in range(episodes):
        grid.reset()
        ct =0
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
            #if we have a special human agent, use it
            if h_agent is None:
                action = np.random.choice(actions, 1)[0]
                h_action = action
            else:
                action = h_agent.get_action(state_ind)
                h_action = action
            #print(action)
            if coagenthelp:
                q_vals = coagent.get_q_values(state_ind)
                #print(q_vals)
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
                #iota = 0.25
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
                #select better closest action if difference between best q and human q is past some threshold
                elif mode == "q-diff" and (human_q-min_q)<(1-threshold*(1-iota**actionct))*(best_q-min_q):
                    actionct += 1
                    #print(action)
                    #print(best_action)
                    #print(state)
                    for i in range(1,5):
                        radial_a = a_dist_ind[action]
                        ccw_ra = (radial_a - i) % len(actions)
                        ccw_a = a_radial[ccw_ra]
                        #print(ccw_a)
                        if q_vals[ccw_a]-min_q >= (1-threshold*(1-iota**actionct))*(best_q-min_q):
                            action = ccw_a
                            break
                        cw_ra = (radial_a + i) % len(actions)
                        cw_a = a_radial[cw_ra]
                        #print(cw_a)
                        if q_vals[cw_a]-min_q >= (1-threshold*(1-iota**actionct))*(best_q-min_q):
                            action = cw_a
                            break
                    #print(action)

                    """a_li = closest_a[action]
                    prev_a_li = []
                    #if the immediate closest actions are not better check the next pair of closest actions
                    #needs to be optimzed as it takes a while to run
                    print(action)
                    while q_vals[action] <= (1-threshold)*best_q:#human_q:
                        print(action)
                        print(a_li)
                        action = a_li[np.argmax([q_vals[a] for a in a_li])]
                        prev_a_li.append(action)
                        #if q_vals[action] <= human_q:
                        new_a_li = []
                        for a in a_li:
                            new_a = closest_a[a]
                            for a1 in new_a:
                                if a1 not in prev_a_li:
                                    new_a_li.append(a1)
                        a_li = new_a_li.copy()"""
            """print(action)
            print(grid.state)
            prev_state = grid.state
            prev_action = action"""
            grid.step(action)
            #print(grid.state)
            if h_agent is not None:
                heatmap[grid.state[1],grid.state[0]] += 1
                new_state = grid.state
                new_state_ind = new_state[1]*(width+1)+new_state[0]
                #print(new_state)
                if (new_state_ind not in [0,1,2,3,10,17,24,23,22,21,28,35,42,43,44,45,46,47,48]):
                    misstepct +=1
            ct+=1
        liRewards.append(grid.reward)
        liRewardsPenalized.append(grid.reward-actionct)#*(1-(actionct/grid.numSteps)))
        actionsctli.append(actionct)#/grid.numSteps)
    rewards = np.array(liRewards)
    rewards_penalized = np.array(liRewardsPenalized)
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
            plt.imshow(heatmap, cmap='cool', interpolation='nearest')
            plt.savefig('two.png')

    return rewards, actionsctli, rewards_penalized

def rl_alone(episodes = 1000, fill_table = False):
    """
    rl agent trained with q-learning for gridworld
    can use fill-table to create a q-table by randomly taking actions
    Returns the trained agent
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
            if fill_table:
                actions = [0, 1, 2, 3, 4, 5, 6, 7]
                action = np.random.choice(actions, 1)[0]
            else:
                action = agent.get_action(state_ind)
            next_state, reward, is_end = grid.step(action)
            next_state_ind = next_state[1]*(width+1)+next_state[0]
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

def q_human(episodes = 1000, fill_table = False):
    """
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

def inverserlQAgent(episodes=1000, mode='override', verbose = True, threshold=1.0):
    q_table = np.load(file="inverserl/gridworld/maxent/results/maxent_q_table.npy")
    path = os.getcwd()
    grid = AdvGridworld()
    os.chdir(path)
    print(grid.name + ' with Random Human '+' and IRL CoAgent('+ mode +')')

    liRewards = []
    liRewardsPenalized = []
    actionsctli =[]
    for i in range(episodes):
        grid.reset()
        ct =0
        actionct = 0
        misstepct = 0
        while (not grid.isEnd) and (grid.numSteps <=200):
            state = grid.state
            width = grid.getBoardDim()[0]
            state_ind = state[1]*(width+1)+state[0]

            if 'disjoint' in mode:
                actions = [0, 1, 2, 3]
            else:
                #'up','right','down','left','upri','dori','dole','uple'
                actions = [0, 1, 2, 3, 4, 5, 6, 7]
            #random human agent
            action = np.random.choice(actions, 1)[0]
            h_action = action


            q_vals = q_table[state_ind]

            #print(q_vals)
            best_action = np.argmax(q_table[state_ind])
            """print("s: ",state, state_ind)
            print("a: ",action, q_table[state_ind][action])
            print("ba: ",best_action, q_table[state_ind][best_action])"""
            best_q = q_vals[best_action]
            min_q = min(q_vals)
            human_q = q_vals[action]
            #print((human_q-min_q)<(1-threshold)*(best_q-min_q))

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
            #select better closest action if difference between best q and human q is past some threshold
            elif mode == "q-diff" and (human_q-min_q)<(1-threshold)*(best_q-min_q):
                actionct += 1
                #print(action)
                #print(best_action)
                #print(state)
                for i in range(1,5):
                    radial_a = a_dist_ind[action]
                    ccw_ra = (radial_a - i) % len(actions)
                    ccw_a = a_radial[ccw_ra]
                    #print(ccw_a)
                    if q_vals[ccw_a]-min_q >= (1-threshold)*(best_q-min_q):
                        action = ccw_a
                        break
                    cw_ra = (radial_a + i) % len(actions)
                    cw_a = a_radial[cw_ra]
                    #print(cw_a)
                    if q_vals[cw_a]-min_q >= (1-threshold)*(best_q-min_q):
                        action = cw_a
                        break
            #print("nba: ",action)

            grid.step(action)
            #print(grid.state)
            ct+=1

        liRewards.append(grid.reward)
        liRewardsPenalized.append(grid.reward-actionct/grid.numSteps)#*(1-(actionct/grid.numSteps)))
        actionsctli.append(actionct/grid.numSteps)#/grid.numSteps)
    rewards = np.array(liRewards)
    rewards_penalized = np.array(liRewardsPenalized)
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

    return rewards, actionsctli, rewards_penalized

def shared_auto_irl():
    q_table = np.load(file="inverserl/gridworld/maxent/results/maxent_q_table.npy")
    path = os.getcwd()
    grid = AdvGridworld()
    os.chdir(path)

    liRewards = []
    liRewardsPenalized = []
    actionsctli =[]
    for i in range(episodes):
        grid.reset()
        ct =0
        actionct = 0
        misstepct = 0
        while (not grid.isEnd) and (grid.numSteps <=200):
            state = grid.state
            width = grid.getBoardDim()[0]
            state_ind = state[1]*(width+1)+state[0]

            actions = [0, 1, 2, 3, 4, 5, 6, 7]
            #random human agent
            action = np.random.choice(actions, 1)[0]
            h_action = action

            q_vals = q_table[state[1]]

            #print(q_vals)
            best_action = np.argmax(q_table[state_ind])
            print(state_ind)
            print(best_action)
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
            #select better closest action if difference between best q and human q is past some threshold
            elif mode == "q-diff" and (human_q-min_q)<(1-threshold)*(best_q-min_q):
                actionct += 1
                #print(action)
                #print(best_action)
                #print(state)
                for i in range(1,5):
                    radial_a = a_dist_ind[action]
                    ccw_ra = (radial_a - i) % len(actions)
                    ccw_a = a_radial[ccw_ra]
                    #print(ccw_a)
                    if q_vals[ccw_a]-min_q >= (1-threshold)*(best_q-min_q):
                        action = ccw_a
                        break
                    cw_ra = (radial_a + i) % len(actions)
                    cw_a = a_radial[cw_ra]
                    #print(cw_a)
                    if q_vals[cw_a]-min_q >= (1-threshold)*(best_q-min_q):
                        action = cw_a
                        break
                #print(action)

            grid.step(action)
            #print(grid.state)
            ct+=1


        liRewards.append(grid.reward)
        liRewardsPenalized.append(grid.reward-actionct/grid.numSteps)#*(1-(actionct/grid.numSteps)))
        actionsctli.append(actionct/grid.numSteps)#/grid.numSteps)
    rewards = np.array(liRewards)
    rewards_penalized = np.array(liRewardsPenalized)
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

    return rewards, actionsctli, rewards_penalized
if __name__ == "__main__":
    """_, h_agent = q_human(episodes=1000, fill_table=True)
    print()
    for y in range(7):
        for x in range(7):
            state_ind = y*(7)+x
            q = h_agent.get_q_values(state_ind)
            print(state_ind, [x,y], np.around(q, decimals=5))
    #grid_human_co(h_agent=h_agent)"""

    """_, rl_agent = rl_alone(1000, fill_table = True)
    print()
    for y in range(7):
        for x in range(7):
            state_ind = y*(7)+x
            q = rl_agent.get_q_values(state_ind)
            print(state_ind, [x,y], np.around(q, decimals=5))
    """
    in_li = []
    r_li = []
    rp_li=[]
    q_table = np.load(file="inverserl/gridworld/maxent/results/maxent_q_table.npy")


    for i in np.arange(0.0, 1.025, 0.025):
        rewards, actionsct, rewardsP = inverserlQAgent(episodes=1000, threshold=i, mode='q-diff')
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
    for i in range(49):
        print(i, q_table[i])
    plt.figure()
    plt.plot(np.arange(0.0, 1.025, 0.025), in_li)
    plt.title('IRL Shared Autonomy Interventions')
    plt.xlabel(r"$\alpha$")
    plt.ylabel('Average intervention count')
    plt.savefig("irl_interventionsr.png")

    plt.figure()
    plt.title('IRL Shared Autonomy Returns', fontsize=14)
    plt.xlabel(r"$\alpha$", fontsize=13)
    plt.ylabel('Average return', fontsize=13)
    plt.plot(np.arange(0.0, 1.025, 0.025), r_li,label='Average environment return')
    plt.plot(np.arange(0.0, 1.025, 0.025), rp_li,label="Average intervention return")
    plt.legend(fontsize=9)
    plt.savefig("irl_rewardsr.png")

    """
    print()
    for iota in np.arange(0.0, 1.1, 0.1):
        in_li = []
        r_li = []
        rp_li=[]
        print("iota: ", iota)
        for i in np.arange(0.0, 1.01, 0.01):
            print("Threshold: ", i)
            rewards, actionsct, rewardsP = grid_human_co(episodes=1000,coagent=rl_agent, threshold=i, verbose=False, mode='q-diff',iota=iota)

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
        plt.plot(np.arange(0.0, 1.01, 0.01), in_li)
        plt.title('Interventions for Shared Autonomy Co-Pilot with Iota=' +str(iota), fontsize=14)
        plt.xlabel(r"$\alpha$", fontsize=13)
        plt.ylabel("Num Interventions/Num Steps", fontsize=13)
        plt.savefig("qmoverinterventionsiota"+str(iota)+".png")

        plt.figure()
        plt.plot(np.arange(0.0, 1.01, 0.01), r_li,label='Return of environment')
        plt.plot(np.arange(0.0, 1.01, 0.01), rp_li,label="Return*(1-(NumInterventions/NumSteps))")
        plt.title('Returns for Shared Autonomy Co-Pilot with Iota=' +str(iota), fontsize=14)
        plt.xlabel(r"$\alpha$", fontsize=13)
        plt.ylabel('Average return', fontsize=13)
        plt.legend(fontsize=9)
        plt.savefig("qmdiffrewardsiota"+str(iota)+".png")

    #grid_human_co(h_agent=h_agent,coagent=rl_agent,mode='q-neg')
    print()
    """
