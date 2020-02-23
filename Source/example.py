import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.gridworld import Gridworld
from systemrl.agents.sarsa import SARSA
from systemrl.agents.sarsa_lambda import SARSALambda
from systemrl.agents.q_learning import QLearning
from systemrl.agents.q_lambda import QLambda

#decaying epsilon method for epsilon-greedy poilcy
def decaying_epsilon(current_episode, total_episodes):
    power = 9*(current_episode/total_episodes)-5
    epsilon = 1/(1+np.exp(power))
    return epsilon


#environment
env = Gridworld()

num_actions = 4
gamma = 0.99
learning_rate = 0.01
lmbda = 0.3

#agent: Note that lambda is not required for SARSA and QLearning
agent = QLambda(num_actions, gamma, learning_rate, lmbda)

episode_returns = []
episode_lengths = []

#total number of episodes
num_episodes = 100

for episodes in range(num_episodes):
    env.reset()
    state = env.state
    is_end = False
    returns = 0
    count = 0
    epsilon = decaying_epsilon(episodes, num_episodes)

    #iterate until episode ends
    while not is_end:
        count += 1
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = agent.get_action(state)
        next_state, reward, is_end = env.step(action)
        agent.train(state, action, reward, next_state)
        returns += reward
        state = next_state
    
    agent.reset()
    episode_returns.append(returns)
    episode_lengths.append(count)

#computing moving averages for returns
window = 10
avg_returns = [np.mean(episode_returns[i:i+window]) for i in range(num_episodes-window)]
avg_returns = [avg_returns[0]]*window + avg_returns

#plot
plt.plot(np.arange(num_episodes), episode_returns, label='returns')
plt.plot(np.arange(num_episodes), avg_returns, label='moving avg')
#note that maximum return is not equivalent to returns for an optimal policy..
#..this is the maximum return observed in an episode
plt.plot(np.arange(num_episodes), [np.max(episode_returns)]*num_episodes, label='maximum return')
plt.legend()
plt.show()
