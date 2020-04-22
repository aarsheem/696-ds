import numpy as np
from systemrl.agents.q_learning import QLearning
from systemrl.agents.adv_nq_learning import NQLearning
from systemrl.agents.nq_learning_old import NQLearningOld
from helper import decaying_epsilon
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_interventions(env, human_policy, agent, num_episodes=500, \
        max_steps=1000, is_update=False):
    episode_returns = []
    interventions = []
    for episodes in range(num_episodes):
        env.reset()
        rawState = env.state
        state_ind = rawState[0][1] * (10) + rawState[0][0]
        state = str(state_ind) + "," + str(rawState[1])
        is_end = False
        inter = 0
        returns = 0
        count = 0
        while not is_end and count < max_steps:
            count += 1
            action = agent.get_action(state)
            human_action = human_policy[state]
            if human_action != action:
                inter += 1
            next_state, reward, is_end = env.step(action)
            if is_update:
                agent.update(reward, is_end)
            rawState = next_state
            state_ind = rawState[0][1] * (10) + rawState[0][0]
            state = str(state_ind) + "," + str(rawState[1])
            returns += reward
        episode_returns.append(returns)
        interventions.append(inter)
    return interventions, episode_returns


def nqcost_old(env, human_policy, min_performance, q2_star, s0, num_episodes, lr1, lr2):
    agent = NQLearningOld(q2_star, lr1, lr2, s0, threshold)
    for episodes in tqdm(range(num_episodes)):
        env.reset()
        state = env.state
        is_end = False
        count = 0
        return1 = 0
        return2 = 0
        epsilon = decaying_epsilon(episodes, num_episodes)
        while not is_end and count < 1000:
            count += 1
            action = agent.get_action(state, epsilon) 
            reward1 = 0 if action == human_policy[state] else -1
            next_state, reward2, is_end = env.step(action)
            agent.train(state, action, reward1, reward2, next_state)
            state = next_state
    return evaluate_interventions(env, human_policy, agent)

def nqcost(env, human_policy, q2_star, gamma, lr, min_performance, num_episodes=100,\
        max_steps=1000):
    R2 = []
    R1 = []
    agent = NQLearning(q2_star, gamma, lr, min_performance)
    for episodes in range(num_episodes):
        env.reset()
        rawState = env.state
        conv_state = rawState[0][1] * (10) + rawState[0][0]
        state = str(conv_state) + "," + str(rawState[1])
        is_end = False
        count = 0
        returns1 = 0
        returns2 = 0
        epsilon = decaying_epsilon(episodes, num_episodes)
        while not is_end and count < max_steps:
            count += 1
            if np.random.random() < epsilon:
                action = np.random.randint(agent.num_actions)
            else:
                action = agent.get_action(state)
            rawNext_state, reward2, is_end = env.step(action)
            if action == human_policy[state]:
                reward1 = 0
            else:
                reward1 = -1
            conv_next_state = rawNext_state[0][1] * 10 + rawNext_state[0][0]
            next_state = str(conv_next_state) + "," + str(rawNext_state[1])
            agent.train(state, action, reward1, reward2, next_state, is_end)
            state = next_state
            returns2 += reward2
            returns1 += reward1
        R1.append(returns1)
        R2.append(returns2)
    """
    X = np.arange(num_episodes)
    plt.plot(X, R2, label="environment return")
    plt.plot(X, np.array(R1) , \
            label="rescaled interventions")
    plt.plot(X, [min_performance]*num_episodes)
    plt.legend()
    plt.show()
    """
    return evaluate_interventions(env, human_policy, agent, is_update=True)

