from ddpg_torch import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
import time

global human_agent_action

env = gym.make('LunarLanderContinuous-v2')
max_timesteps = 3000

UP_ARROW =  65362
DOWN_ARROW = 65364
LEFT_ARROW = 65361
RIGHT_ARROW = 65363


human_agent_action = [0, 0]
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action
    if key == UP_ARROW:
        human_agent_action[0] = 1
    elif key == LEFT_ARROW:
        human_agent_action[1] = max(-1, human_agent_action[1] - 1)
    elif key == RIGHT_ARROW:
        human_agent_action[1] = min(1, human_agent_action[1] + 1)
    else:
        return

def key_release(key, mod):
    global human_agent_action
    if key == UP_ARROW:
        human_agent_action[0] = 0
    elif key == LEFT_ARROW:
        human_agent_action[1] = min(1, human_agent_action[1] + 1)
    elif key == RIGHT_ARROW:
        human_agent_action[1] = max(-1, human_agent_action[1] - 1)
    else:
        return

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release



ver_agent = Agent(name="ver_agent",alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,batch_size=64,  layer1_size=400, layer2_size=300, n_actions=1)
hor_agent = Agent(name="hor_agent",alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

ver_agent.load_models(0)
hor_agent.load_models(0)

np.random.seed(0)

score_history = []



for i in range(2000):
    obs = env.reset()
    done = False
    score = 0
    for _ in range(max_timesteps):
        ver_act = np.round(ver_agent.choose_action(obs))
        hor_act = np.round(hor_agent.choose_action(obs))
    
        
        act = [ver_act[0],hor_act[1]-hor_act[0]] # Bot Controls Everything
        #act = [ver_act[0],human_agent_action[1]] # Bot Controls Up

        #act = [human_agent_action[0],hor_act[1]-hor_act[0]] # Bot Controls Right Left
    
        new_state, reward, done, info = env.step(act)
        
        score += reward
        obs = new_state
        if done:
            break
        env.render()
        time.sleep(0.03)
    score_history.append(score)
    

    print("Score: %0.2f" % (score))
