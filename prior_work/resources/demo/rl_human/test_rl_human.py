from ddpg_torch import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
import time
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
        
score = [0,0,0,0,0]

#ver_agent = Agent(name="ver_agent",alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,batch_size=64,  layer1_size=400, layer2_size=300, n_actions=1)
hor_agent = Agent(name="hor_agent",alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

#model_vert = load_model('../../../models/lunar_lander/lunar_lander_dqn_hori_v2_800.h5')
#model_hori = load_model('../../../models/lunar_lander/lunar_lander_dqn_vert_v2_800.h5')
#ver_agent.load_models(0)
hor_agent.load_models(0)

def rollout(env):
    global human_agent_action
    env.reset()
    total_reward = 0
    total_timesteps = 0
    obser = [0,0,0,0,0,0,0,0]
    while 1:
        #action = list(np.round(model.predict(np.array([obser])))[0])
        hor_act = np.round(hor_agent.choose_action(obser))
        action_env_space = [human_agent_action[0],hor_act[1]-hor_act[0]]
        #action_env_space = [human_agent_action[0],action[2]-action[1]]
        total_timesteps += 1

        obser, r, done, info = env.step(action_env_space)

        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done:
            break
        
        time.sleep(0.035)
    
    score.pop(0)
    score.append(total_reward)
    print("Score %i \t Last 5 Average %0.2f" % (total_reward, sum(score)/5))



while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break