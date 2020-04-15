import numpy as np

def best_action_min_value(q, state, DP):
    if state in DP:
        return DP[state]
    max_value = -np.inf
    min_value = np.inf
    for action,_ in enumerate(q[state]):
        if q[state][action] == max_value:
            possible_actions.append(action)
        elif q[state][action] > max_value:
            max_value = q[state][action]
            possible_actions = [action]
        min_value = min(min_value, q[state][action])
    DP[state] = [possible_actions, min_value]  
    return DP[state]

    
def shared_autonomy(env, human_policy, q_star, alpha, num_episodes=100, max_steps=1000):
    DP = {}
    episode_returns = []
    interventions = []
    for i in range(num_episodes):
        env.reset()
        state = env.state
        is_end = False
        inter = 0
        returns = 0
        count = 0
        while not is_end and count < max_steps:
            count += 1
            human_action = human_policy[state]
            possible_actions, min_value = best_action_min_value(q_star, state, DP)
            best_action = np.random.choice(possible_actions)
            q_human = q_star[state][human_action] - min_value
            q_best = q_star[state][best_action] - min_value
            if q_human >= (1-alpha)*(q_best):
                action = human_action
            else:
                action = best_action
                inter += 1
            state, reward, is_end = env.step(action)
            returns += reward
        episode_returns.append(returns)
        interventions.append(inter)
    return np.mean(interventions), np.mean(episode_returns)
