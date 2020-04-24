import numpy as np
from systemrl.agents.q_learning import QLearning
from systemrl.agents.nq_learning import NQLearning
from systemrl.agents.nq_learning_old import NQLearningOld
from helper import decaying_epsilon, evaluate_interventions
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def nqcost(env, human_policy, q2_star, gamma, lr, min_performance, num_episodes=100, max_steps=1000, return_dict=None):
    R2 = []
    R1 = []
    agent = NQLearning(q2_star, gamma, lr, min_performance)
    for episodes in range(num_episodes):
        env.reset()
        state = env.state
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
            next_state, reward2, is_end = env.step(action)
            if action == human_policy[state]:
                reward1 = 0
            else:
                reward1 = -1
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
    ans = evaluate_interventions(env, human_policy, agent, 500, 1000, True)
    if return_dict is not None:
        return_dict[min_performance, lr] = ans
    return ans

