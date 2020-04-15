import numpy as np
from systemrl.agents.q_learning import QLearning
import matplotlib.pyplot as plt
from helper import decaying_epsilon, evaluate_interventions
from tqdm import tqdm

#This is not converging
def mincost(env, human_policy, min_performance, agent, num_episodes=1000, max_steps=1000):
    print("mincost")
    episode_returns_ = []
    for episodes in tqdm(range(num_episodes)):
        env.reset()
        state = env.state
        is_end = False
        returns = 0
        returns_ = 0
        count = 0
        epsilon = decaying_epsilon(episodes, num_episodes)
        while not is_end and count < max_steps:
            count += 1
            if np.random.random() < epsilon:
                action = np.random.randint(agent.num_actions)
            else:
                action = agent.get_action(state)
            next_state, reward, is_end = env.step(action)
            returns += reward
            if is_end == True or count >= max_steps:
                if returns < min_performance:
                    reward_ = -100
            elif action == human_policy[state]:
                reward_ = 0
            else:
                reward_ = -1
            agent.train(state, action, reward_, next_state)
            state = next_state
            returns_ += reward_
        episode_returns_.append(returns_)
    return evaluate_interventions(env, human_policy, agent)

        

    
