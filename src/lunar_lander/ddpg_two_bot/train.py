from ddpg_torch import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np



env = gym.make('LunarLanderContinuous-v2')
max_timesteps = 300


ver_agent = Agent(name = "ver_agent",alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,batch_size=64,  layer1_size=400, layer2_size=300, n_actions=1)
hor_agent = Agent(name = "hor_agent",alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)


#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(10000):
    obs = env.reset()
    done = False
    score = 0
    for _ in range(max_timesteps):
        ver_act = np.round(ver_agent.choose_action(obs))
        hor_act = np.round(hor_agent.choose_action(obs))
        
        act = [ver_act[0],hor_act[1]-hor_act[0]]
    
        new_state, reward, done, info = env.step(act)

        hor_agent.remember(obs, hor_act, reward, new_state, int(done))
        ver_agent.remember(obs, ver_act, reward, new_state, int(done))

        hor_agent.learn()
        ver_agent.learn()

        score += reward
        obs = new_state

        if done:
            break
        #env.render()
    score_history.append(score)

    if i % 50 == 0:
        hor_agent.save_models()
        ver_agent.save_models()


    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
