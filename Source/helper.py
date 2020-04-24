import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def decaying_epsilon(current_episode, total_episodes):
    power = 9*(current_episode/total_episodes)-5
    epsilon = 1/(1+np.exp(power))
    return epsilon

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
            action = policy[state]
            next_state, reward, is_end = env.step(action)
            count += 1
            state = next_state
            rewards += reward
        returns.append(rewards)
    return returns

def optimal_agent(env, agent, num_episodes=100, max_steps=1000):
    print("optimal agent")
    for episodes in tqdm(range(num_episodes)):
        env.reset()
        state = env.state
        is_end = False
        count = 0
        epsilon = decaying_epsilon(episodes, num_episodes)
        while not is_end and count < max_steps:
            count += 1
            if np.random.random() < epsilon:
                action = np.random.randint(agent.num_actions)
            else:
                action = agent.get_action(state)
            next_state, reward, is_end = env.step(action)
            #if not able to reach end get a large negative reward
            if count >= max_steps:
                reward = -100
            agent.train(state, action, reward, next_state)
            state = next_state
    return agent

def evaluate_interventions(env, human_policy, agent, num_episodes=500, \
        max_steps=1000, is_update=False):
    episode_returns = []
    interventions = []
    for episodes in range(num_episodes):
        env.reset()
        state = env.state
        is_end = False
        inter = 0
        returns = 0
        count = 0
        while not is_end and count < max_steps:
            count += 1
            human_action = human_policy[state]
            if is_update:
                action = agent.check_action(state,human_action)
            else:
                action = agent.get_action(state)
            if human_action != action:
                inter += 1
            next_state, reward, is_end = env.step(action)
            if is_update:
                agent.update(reward, is_end)
            state = next_state
            returns += reward
        episode_returns.append(returns)
        interventions.append(inter)
    return interventions, episode_returns
