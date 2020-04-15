import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from helper import decaying_epsilon, save_obj, load_obj
from helper import evaluate_policy, optimal_agent
from systemrl.environments.gridworld import Gridworld
from systemrl.agents.q_learning import QLearning
from systemrl.agents.sarsa import SARSA
from systemrl.agents.sarsa_lambda import SARSALambda
from systemrl.agents.q_lambda import QLambda
from shared_autonomy import shared_autonomy
from shared_modified import shared_modified
from mincost import mincost
from nqcost import nqcost
from copy import deepcopy
import multiprocessing


def init_random_policy(num_states, num_actions):
    policy = [np.random.choice(num_actions) for i in range(num_states)]
    return policy

def init_nonrandom_policy():
    (u,d,l,r) = (0, 1, 2, 3)
    human_policy = [
            r, r, r, r, d,
            u, d, l, l, l,
            d, l, l, r, d,
            d, r, r, u, d, 
            r, r, u, u, u]
    return human_policy

#-----environment------
np.random.seed(0)
startState=0
endStates=[24]
shape=[5,5]
obstacles=[]
waterStates=np.arange(1,24)
waterRewards=[-1,-1,-1,-5,-9,-3,-4,-1,-6,-5,-1,-1,-1,-2,-6,-1,-8,-4,-5,-7,-1,-1,-1]
env = Gridworld(startState, endStates, shape, obstacles, waterStates, waterRewards)
#----------------------

gamma = 1
num_actions = 4
num_states = shape[0]*shape[1]
human_policy = init_nonrandom_policy()#init_random_policy(25, 4)
human_performance = np.mean(evaluate_policy(env, human_policy))
print("human performance: ", human_performance)
saved_model = True

if not saved_model:
    learning_rate = 0.005
    untrained_agent = QLearning(num_actions, gamma, learning_rate)
    trained_agent = optimal_agent(env, untrained_agent, num_episodes=10000)
    q_star = trained_agent.q_table
    save_obj(dict(q_star), "q_star")
else:
    q_star = load_obj("q_star")
J_star = np.max(q_star[startState])
baseline = human_performance

shared = []
nq_cost = []
min_cost = []
mod_shared = []

alphas = np.concatenate((np.arange(0, 1e-3, 1e-4), np.arange(1e-3, 1e-2, 1e-3), np.arange(1e-2, 0.1, 1e-2), np.arange(0.1, 1.01, 0.1)))
for alpha in tqdm(alphas):
    min_performance = baseline + alpha * (J_star-baseline)
    #mincost
    """
    untrained_agent = QLambda(num_actions, gamma, learning_rate, 0.5)
    interventions, returns = mincost(env, human_policy, min_performance, untrained_agent, num_episodes=1000)
    """

    #shared autonomy
    interventions, returns = shared_autonomy(env, human_policy, q_star, alpha)
    shared.append((np.mean(interventions), np.mean(returns)))
    
    #modified shared autonomy
    interventions, returns = shared_modified(env, human_policy, q_star, gamma, min_performance)
    mod_shared.append((np.mean(interventions), np.mean(returns)))

    #nqlearning
    lr = 0.01
    interventions, returns = nqcost(env, human_policy, deepcopy(q_star), gamma,\
            lr, min_performance, num_episodes = 1000)
    nq_cost.append((np.mean(interventions),np.mean(returns)))

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



