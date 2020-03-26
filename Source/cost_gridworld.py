import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.gridworld import Gridworld
from systemrl.agents.sarsa import SARSA
from systemrl.agents.sarsa_lambda import SARSALambda
from systemrl.agents.q_learning import QLearning
from systemrl.agents.q_lambda import QLambda
from tqdm import tqdm

#decaying epsilon method for epsilon-greedy poilcy
def decaying_epsilon(current_episode, total_episodes):
    power = 9*(current_episode/total_episodes)-5
    epsilon = 1/(1+np.exp(power))
    return epsilon

#Actions: up (0), down (1), left (2), right (3), do nothing (4)
human_policy = [3,1,3,3,1,\
                3,3,2,3,1,\
                0,2,0,3,1,\
                0,1,0,1,2,\
                0,3,3,2,0]

def get_performance(env, policy):
    episodes = 100
    episode_returns = []
    for i in range(episodes):
        count = 0
        env.reset()
        state = env.state
        is_end = False
        returns = 0
        while not is_end and count < 1000:
            count += 1
            action = policy[state]
            state, reward, is_end = env.step(action)
            returns += reward 
        episode_returns.append(returns)
    return np.mean(episode_returns)



#environment
env = Gridworld()

num_actions = 4
gamma = 0.99
learning_rate = 0.01
lmbda = 0.3

#agent: Note that lambda is not required for SARSA and QLearning
agent = QLearning(num_actions, gamma, learning_rate)

episode_returns = []
episode_returns_ = []
episode_lengths = []

#total number of episodes
num_episodes = 10000

for episodes in tqdm(range(num_episodes)):
    env.reset()
    state = env.state
    is_end = False
    returns = 0
    returns_ = 0
    count = 0
    epsilon = decaying_epsilon(episodes, num_episodes)
    #iterate until episode ends
    while not is_end:
        count += 1
        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = agent.get_action(state)
        next_state, reward, is_end = env.step(action)
        returns += reward
        if returns < 0 or count > 50:
            reward_ = -100
            is_end = True
        elif action == human_policy[state]:
            reward_ = 0
        else:
            reward_ = -1
        agent.train(state, action, reward_, next_state)
        state = next_state
        returns_ += reward_
    
    agent.reset()
    episode_returns.append(returns)
    episode_returns_.append(returns_)
    episode_lengths.append(count)

#compare current and human policy
diff = 0
agent_policy = agent.get_policy()
for s in agent_policy:
    if agent_policy[s] != human_policy[s]:
        diff += 1
        print("state: {:2d}, human: {}, agent: {}".format(s,\
            human_policy[s], agent_policy[s]))
print("difference: ",diff)

#performance
human_performance = get_performance(env, human_policy)
agent_performance = get_performance(env, agent_policy)

#computing moving averages for returns
window = 10
avg_returns = [np.mean(episode_returns[i:i+window]) for i in range(num_episodes-window)]
avg_returns = [avg_returns[0]]*window + avg_returns

avg_lengths = [np.mean(episode_lengths[i:i+window]) for i in range(num_episodes-window)]
avg_lengths = [avg_lengths[0]]*window + avg_lengths

#plot
plt.plot(np.arange(num_episodes), episode_returns, label='returns')
plt.plot(np.arange(num_episodes), episode_returns_, label='returns_')
plt.plot(np.arange(num_episodes), avg_returns, label='moving avg')
#note that maximum return is not equivalent to returns for an optimal policy..
#..this is the maximum return observed in an episode
plt.plot(np.arange(num_episodes), [np.max(episode_returns)]*num_episodes, label='maximum return')
#plt.plot(np.arange(num_episodes), [human_performance]*num_episodes, label='human performance')
plt.plot(np.arange(num_episodes), [agent_performance]*num_episodes, label='agent performance')
plt.legend()
plt.show()

plt.plot(np.arange(num_episodes), episode_lengths, label='episode_lengths')
plt.plot(np.arange(num_episodes), avg_lengths, label='moving avg')
plt.legend()
plt.show()
