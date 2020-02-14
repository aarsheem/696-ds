# Introduction 
Project AlphaBlue - exploring collaborative agents for game playing

# Getting Started
Get started with the demo agents for lunar lander

Go to /prior_work/resources/demo and cd into desired demo <br>
run `python test.py` 





# Modules

resources - 

1) Demo - contains files for the various lunarlander demos <br>
    1 - baseline_nn: Agent plays by itself, trained using just prior gameplay data <br>
    2 - bot_hori: Agent control left,right , player(human) controls up<br>
    3 - bot_vert: Agent controls up, player controls left, right<br>
    4 - rl_human: Agent control left,right , player(human) controls up ( Agent trained while playing against "human model" (Simple NN that was trained using gamplay data from real players)<br>
    5 - step_in: Agent steps in and takes control if human doesnt provide input for a certain amount of time (Buggy)<br>
    6 - user_full_control: Player has full control<br>

2) gym-cust-cartpole - custom gym environment for cartpole (Allows for No-Op as input)

3) gym-cust-lunarlander - unused

data\lunarlander - 
    Contains recorded gameplay data from real human players
    Schema - [[Environment State at time T], [Input Action by player at time T], [Score at time T]]

models - 
    Model dumps for agents, (ignore lunar_lander models - outdated)


notebooks -
    Notebook for training models
    lunar_dqn_mirror - trains two agents (DQN) (one that controls left, right and one that controls up)
    lunarlander_baselne_regular - trains "human model" from recorded gameplay data
    lunarlander_baseline_rnn - incomplete


src -
    breakout - incomplete
    cartpole -
        cartpole_mirror - trains two agents for cartpole game
        cartpole_prod - agent controls left, human controls right
        cartpole_self - two agent plays by themselves
    lunar lander - 
        ddpg_two_bot - 
            ddpg_torch.py - defines the actor-critic network
            train.py - trains two agent(s) (DDPG) to play lunar lander 
            
    other/
        utils, lunarlander gameplay recorder

    

# Related documentations
OpenAI Gym - https://gym.openai.com/docs/
