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
from multiprocessing import Manager, Process


def init_policy(randomness):
    (u,d,l,r) = (0, 1, 2, 3)
    optimal_policy = [
            r, r, r, d, l,
            r, d, d, d, d,
            r, d, l, l, d,
            r, d, d, d, d, 
            r, r, r, r, u]
    random_states = int(randomness/100*25)
    all_states = np.arange(25)
    np.random.shuffle(all_states)
    selected_states = all_states[:random_states]
    for state in selected_states:
        optimal_policy[state] = np.random.choice(4)
    return optimal_policy

#-----environment------
np.random.seed(0)
startState=0
endStates=[24]
shape=[5,5]
obstacles=[]
waterStates=np.arange(1,24)
waterRewards= [-1,-1,-1,-5,\
            -9,-3,-4,-1,-6,\
            -5,-1,-1,-1,-2,\
            -6,-1,-8,-4,-5,\
            -7,-1,-1,-1]
env = Gridworld(startState, endStates, shape, obstacles, waterStates, waterRewards)
#----------------------

randomness = 20
gamma = 0.95
num_actions = 4
num_states = shape[0]*shape[1]
human_policy = init_policy(randomness)
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


manager = Manager()
return_dict = manager.dict()
jobs = []
for alpha in tqdm(np.arange(0,1,0.05)):
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
    for lr in np.random.sample(20)*0.1:
        p = Process(target=nqcost, args=(env, human_policy, deepcopy(q_star), gamma, lr, min_performance, 1000, 1000, return_dict))
        jobs.append(p)
        p.start()
    #nq_cost.append((np.mean(interventions),np.mean(returns)))
    
for p in jobs:
    p.join()

count = 0
best_interventions = np.inf
best_returns = -np.inf
for key in sorted(return_dict):
    count += 1
    interventions = np.mean(return_dict[key][0])
    returns = np.mean(return_dict[key][1])
    if interventions < best_interventions:
        best_interventions = interventions
        best_returns = returns
    if count == 20:
        nq_cost.append((best_interventions, best_returns))
        count = 0
        best_interventions = np.inf
        best_returns = -np.inf


nq_cost = np.array(nq_cost).T
shared = np.array(shared).T
mod_shared = np.array(mod_shared).T

print(shared)
print(nq_cost)

plt.scatter(mod_shared[0], mod_shared[1], label="mod shared")
plt.scatter(shared[0], shared[1], label="shared")
plt.scatter(nq_cost[0], nq_cost[1], label="nq")
plt.xlabel("interventions")
plt.ylabel("returns")

plt.legend()
plt.savefig("comparison_"+str(randomness)+".png")
#plt.show()



