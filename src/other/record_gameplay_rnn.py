import sys, gym, time
import os
import json


if not os.path.exists('gameplay_rec_rnn'):
    os.makedirs('gameplay_rec_rnn')

UP_ARROW =  65362
DOWN_ARROW = 65364
LEFT_ARROW = 65361
RIGHT_ARROW = 65363


env = gym.make('LunarLanderContinuous-v2')
print("Action Space : ",env.action_space)
print("Observation Space : ",env.observation_space)

gameplay_file = open(os.path.join("gameplay_rec_rnn","lunar_gameplay_rnn_"+str(time.time())+".tsv"),"w")

human_agent_action = [0, 0]
left_button = 0
right_button = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action,left_button,right_button
    if key == UP_ARROW:
        human_agent_action[0] = 1
    elif key == LEFT_ARROW:
        left_button = 1
        human_agent_action[1] =  right_button - left_button
    elif key == RIGHT_ARROW:
        right_button = 1
        human_agent_action[1] = right_button - left_button
    else:
        return

def key_release(key, mod):
    global human_agent_action,left_button,right_button
    if key == UP_ARROW:
        human_agent_action[0] = 0
    elif key == LEFT_ARROW:
        left_button = 0
        human_agent_action[1] = right_button - left_button
    elif key == RIGHT_ARROW:
        right_button = 0
        human_agent_action[1] = right_button - left_button
    else:
        return

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action,left_button,right_button
    env.reset()
    total_reward = 0
    total_timesteps = 0
    obser, r, done, info = env.step(human_agent_action)
    while 1:
        total_timesteps += 1
        out_payload = [list(obser),[int(human_agent_action[0]),int(left_button),int(right_button)],float(total_reward)]
        gameplay_file.write(str(out_payload))
        gameplay_file.write("\n")
        obser, r, done, info = env.step(human_agent_action)

        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done:
            break
        
        time.sleep(0.04)
    
    gameplay_file.write("---")
    gameplay_file.write("\n")
    print("timesteps %i score %0.2f" % (total_timesteps, total_reward))



while 1:
    window_still_open = rollout(env)
    if window_still_open==False:
        gameplay_file.close()
        break





###########
# Observation Vector = [ NO-OP , UP , LEFT RIGHT]
#
#
##########