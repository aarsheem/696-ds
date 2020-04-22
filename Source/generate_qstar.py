from helper import optimal_agent
from systemrl.agents.q_learning import QLearning
from systemrl.environments.advgridworld import AdvGridworld
import pickle


num_actions = 6
gamma = 1
env = AdvGridworld(2)

learning_rate = 0.005
untrained_agent = QLearning(num_actions, gamma, learning_rate)
trained_agent = optimal_agent(env, untrained_agent, num_episodes=10000)
q_star = trained_agent.q_table

with open('q_star_advgridworld2.pkl', 'wb') as f:
    pickle.dump(dict(q_star), f, pickle.HIGHEST_PROTOCOL)