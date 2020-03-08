import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.advgridworld import AdvGridworld
from systemrl.agents.sarsa import SARSA
from systemrl.agents.sarsa_lambda import SARSALambda
from systemrl.agents.q_learning import QLearning
from systemrl.agents.q_lambda import QLambda
from tqdm import tqdm
from collections import defaultdict

#decaying epsilon method for epsilon-greedy poilcy
def decaying_epsilon(current_episode, total_episodes):
    power = 9*(current_episode/total_episodes)-5
    epsilon = 1/(1+np.exp(power))
    return epsilon

def human_policy(human_actions, action_index):
    possible_actions = [action_index[i] for i in human_actions]
    human = defaultdict(lambda: -1)
    best_action = [2,1,1,2,3,3,3,\
            2,4,4,2,7,7,7,\
            2,4,4,2,7,7,7,\
            2,1,1,2,3,3,2,\
            5,5,2,2,5,5,2,\
            5,5,5,2,5,5,2,\
            1,1,1,1,1,1,1]
    for idx, action in enumerate(best_action):
        if action in possible_actions:
            human[idx] = action
    return human

def get_state(state):
    return state[0]+state[1]*7

    
human_actions = ["right","left","upri","dori"]
agent_actions = ["up", "down","uple","dole"]


action_index = {"up":0,"right":1,"down":2,"left":3,"upri":4,"dori":5,"dole":6,"uple":7}
env = AdvGridworld()
human = human_policy(human_actions, action_index)
num_actions = len(agent_actions) + 1 # +1 for letting human take action
gamma = 0.9
learning_rate = 0.01
lmbda = 0.3

#agent: Note that lambda is not required for SARSA and QLearning
agent = QLearning(num_actions, gamma, learning_rate)

episode_returns = []
episode_lengths = []

#total number of episodes
num_episodes = 1000
#maximum number of steps per episode. if agent exceeds this..
#..we terminate and give a large negative reward
max_count = 200

for episodes in tqdm(range(num_episodes)):
    env.reset()
    state = env.state
    _state = get_state(state)
    is_end = False
    returns = 0
    count = 0
    epsilon = decaying_epsilon(episodes, num_episodes)

    #iterate until episode ends
    while not is_end and count<max_count:
        count += 1
        #selection of agent action 
        if np.random.random() < epsilon:
            agent_action = np.random.randint(num_actions)
        else:
            agent_action = agent.get_action(_state)
        #human is taking action
        if agent_action == num_actions-1:
            action = human[_state]
            #we do not have option for no movement here..
            #..if the human has no good action here, she..
            #will take any random action
            if action == -1:
                action = action_index[np.random.choice(human_actions)]
        #agent is taking action
        else:
            action = action_index[agent_actions[agent_action]]
        next_state, reward, is_end = env.step(action)
        #large negative reward for exceeding steps limit
        if count == max_count:
            reward = -100
        _state = get_state(state)
        _next_state = get_state(next_state)
        agent.train(_state, agent_action, reward, _next_state)
        returns += reward
        state = next_state
    
    agent.reset()
    episode_returns.append(returns)
    episode_lengths.append(count)



#computing moving averages for returns
window = 20 #should be small compared to total episodes
avg_returns = [np.mean(episode_returns[i:i+window]) for i in range(num_episodes-window)]
avg_returns = [avg_returns[0]]*window + avg_returns
print("Performance:", avg_returns[-1])
#plot
plt.plot(np.arange(num_episodes), episode_returns, label='returns')
plt.plot(np.arange(num_episodes), avg_returns, label='moving avg')
#note that maximum return is not equivalent to returns for an optimal policy..
#..this is the maximum return observed in an episode
plt.plot(np.arange(num_episodes), [np.max(episode_returns)]*num_episodes, label='maximum return')
plt.legend()
plt.show()
