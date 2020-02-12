import sys, gym, time
from keras.models import load_model
import numpy as np
import pickle

UP_ARROW =  65362
DOWN_ARROW = 65364
LEFT_ARROW = 65361
RIGHT_ARROW = 65363

max_timesteps = 3000
env = gym.make('LunarLanderContinuous-v2')

model = load_model('models/test_model_2.h5')

# model = pickle.load(open('baseline_supervised/forest_model.sav', 'rb'))

env.render()
human_agent_action = [0, 0]

score = [0,0,0,0,0]

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

def rollout(env):
    global human_agent_action
    env.reset()
    total_reward = 0
    total_timesteps = 0
    obser = [0,0,0,0,0,0,0,0]
    for _ in range(max_timesteps):
        action = list(np.round(model.predict(np.array([obser])))[0])
        action_env_space = [action[0],human_agent_action[1]]
        #action_env_space = [human_agent_action[0],action[2]-action[1]]
        total_timesteps += 1

        obser, r, done, info = env.step(action_env_space)

        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done:
            break
        
        time.sleep(0.03)

    score.pop(0)
    score.append(total_reward)
    print("Score %i \t Last 5 Average %0.2f" % (total_reward, sum(score)/5))



while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break
