import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from helper import decaying_epsilon, save_obj, load_obj
from systemrl.environments.gridworld import Gridworld
from systemrl.agents.q_learning import QLearning
from systemrl.agents.sarsa import SARSA
from systemrl.agents.sarsa_lambda import SARSALambda
from systemrl.agents.q_lambda import QLambda
from adv_shared_autonomy import shared_autonomy
from adv_shared_modified import shared_modified
from mincost import mincost
from adv_nqcost import nqcost
from copy import deepcopy
import multiprocessing
from systemrl.environments.advgridworld import AdvGridworld
import pickle
import os
from makeHumanPolicy import getPolicy


def init_random_policy(num_states, num_actions):
    policy = [np.random.choice(num_actions) for i in range(num_states)]
    return policy


def evaluate_policy(env, policy, num_episodes=100, max_steps=1000):
    print("evaluating policy")
    returns = []
    for episodes in tqdm(range(num_episodes)):
        env.reset()
        state = env.state
        is_end = False
        count = 0
        rewards = 0
        while not is_end and count < max_steps:
            conv_state = state[0][1] * (10) + state[0][0]
            stateId = str(conv_state) + "," + str(state[1])
            action = policy[stateId]
            next_state, reward, is_end = env.step(action)
            count += 1
            state = next_state
            rewards += reward
        returns.append(rewards)
    return returns

with open('obj\q_star_advgridworld2.pkl', 'rb') as f:
    q_star = pickle.load(f)

np.random.seed(0)
# -----environment------
env = AdvGridworld(2)
# ----------------------

gamma = 1
num_actions = 6

human_policy = getPolicy()
human_performance = np.mean(evaluate_policy(env, human_policy))
print("human performance: ", human_performance)

startRaw = env.getStart()
startInd = startRaw[0][1] * (10 + 1) + startRaw[0][0]
startState = str(startInd) + "," + str(startRaw[1])
J_star = np.max(q_star[startState])
baseline = human_performance

shared = []
nq_cost = []
min_cost = []
mod_shared = []

alphas = np.concatenate(
    (np.arange(0, 1e-3, 1e-4), np.arange(1e-3, 1e-2, 1e-3), np.arange(1e-2, 0.1, 1e-2), np.arange(0.1, 1.01, 0.1)))
for alpha in tqdm(alphas):
    min_performance = baseline + alpha * (J_star - baseline)
    # mincost
    """
    untrained_agent = QLambda(num_actions, gamma, learning_rate, 0.5)
    interventions, returns = mincost(env, human_policy, min_performance, untrained_agent, num_episodes=1000)
    """

    # shared autonomy
    interventions, returns = shared_autonomy(env, human_policy, q_star, alpha)
    shared.append((np.mean(interventions), np.mean(returns)))

    # modified shared autonomy
    interventions, returns = shared_modified(env, human_policy, q_star, gamma, min_performance)
    mod_shared.append((np.mean(interventions), np.mean(returns)))

    # nqlearning
    lr = 0.01
    interventions, returns = nqcost(env, human_policy, deepcopy(q_star), gamma, \
                                    lr, min_performance, num_episodes=1000)
    nq_cost.append((np.mean(interventions), np.mean(returns)))

nq_cost = np.array(nq_cost).T
shared = np.array(shared).T
mod_shared = np.array(mod_shared).T

plt.plot(mod_shared[0], mod_shared[1], label="mod shared")
plt.plot(shared[0], shared[1], label="shared")
plt.plot(nq_cost[0], nq_cost[1], label="nq")
plt.xlabel("interventions")
plt.ylabel("returns")

plt.legend()
plt.show()