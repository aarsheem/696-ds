import sys, gym, time



UP_ARROW =  65362
DOWN_ARROW = 65364
LEFT_ARROW = 65361
RIGHT_ARROW = 65363


#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = gym.make('LunarLanderContinuous-v2')
print("Action Space : ",env.action_space)
print("Observation Space : ",env.observation_space)


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

def rollout(env):
    global human_agent_action
    env.reset()
    total_reward = 0
    total_timesteps = 0
    while 1:
        a = human_agent_action
        total_timesteps += 1


        obser, r, done, info = env.step(human_agent_action)

        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
          
        if done:
            break
        
        #time.sleep(0.035)
    print("Score %0.2f" % ( total_reward))    



while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break






