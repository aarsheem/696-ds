import sys, gym, time
from keras.models import load_model
import numpy as np
import pickle



env = gym.make('LunarLanderContinuous-v2')

<<<<<<< HEAD
model = load_model('../../models/lunar_lander/test_model.h5')

# model = pickle.load(open('baseline_supervised/forest_model.sav', 'rb'))
=======
model = load_model('models/lunar_lander/test_model.h5')


>>>>>>> cc9c97c52777b6306afa2efa9945c33d52a8dded

env.render()

def rollout(env):

    env.reset()
    total_reward = 0
    total_timesteps = 0
    obser = [0,0,0,0,0,0,0,0]
    while 1:
        action = list(np.round(model.predict(np.array([obser])))[0])
        print(action)
        action_env_space = [action[0],action[2]-action[1]] 
        total_timesteps += 1

        obser, r, done, info = env.step(action_env_space)

        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done:
            break
        
        time.sleep(0.035)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))



while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break
