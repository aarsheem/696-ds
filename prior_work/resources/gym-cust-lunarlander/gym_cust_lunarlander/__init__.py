from gym.envs.registration import register

register(
    id='lunarlandercust-v1',
    entry_point='gym_cust_lunarlander.envs:LunarLanderCust',
)

register(
    id='lunarlandercustcont-v1',
    entry_point='gym_cust_lunarlander.envs:LunarLanderCustContinuous',
)