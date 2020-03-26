import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.gridworld import Gridworld
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

def human_policy(difficulty, human_actions):
    human = defaultdict(lambda: 4)
    if difficulty == "easy":
        up = [5, 10, 15, 20]
        right = [0, 1, 2, 3, 6, 7, 8, 13, 18, 23, 22]
        down = [4, 9, 14, 19]
        left = [11, 16, 21]
    else:
        up = [17,99,89,79,69,59,52,42,32,34,35,36,62,63,72,73,74]
        right = [90,91,92,93,94,95,96,97,98,22,23,24,25]
        left = [27,9,8,7,6,5,4,3,2,11,49,48,77,76,75,64,53,54,55]
        down = [0,1,10,20,30,40,50,60,70,80,39,38,37,47,57,67,66,65]
    if "up" in human_actions:
        for i in up:
            human[i] = 0
    if "down" in human_actions:
        for i in down:
            human[i] = 1
    if "left" in human_actions:
        for i in left:
            human[i] = 2
    if "right" in human_actions:
        for i in right:
            human[i] = 3
    return human

def environment(difficulty):
    if difficulty == "easy":
        return Gridworld()
    start=6
    end=26
    shape=(10,10)
    obstacles=(12,13,14,15,16, 18,19,28,29, 21,31,41,51,61,71,81, \
        82,83,84,85,86,87,88, 78,68,58, 33,43,44,45,46,56)
    bad_states=(0, 17, 27, 36, 55, 72)
    return Gridworld(start, end, shape, obstacles, bad_states)

    
difficulty = "difficult"
human_actions = ["left"]
agent_actions = ["up", "down","right"]


action_index = {"up":0,"down":1,"left":2,"right":3}
env = environment(difficulty)
human = human_policy(difficulty, human_actions)
num_actions = len(agent_actions) + 1 # +1 for letting human take action
gamma = 0.99
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
max_count = 2000

for episodes in tqdm(range(num_episodes)):
    env.reset()
    state = env.state
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
            agent_action = agent.get_action(state)
        
        #human is taking action
        if agent_action == num_actions-1:
            action = human[state]
        #agent is taking action
        else:
            action = action_index[agent_actions[agent_action]]
       
        next_state, reward, is_end = env.step(action)
        #large negative reward for exceeding steps limit
        if count == max_count:
            reward = -100
        agent.train(state, agent_action, reward, next_state)
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
