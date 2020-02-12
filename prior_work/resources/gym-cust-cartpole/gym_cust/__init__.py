from gym.envs.registration import register

register(
    id='cartpolenoop-v1',
    entry_point='gym_cust.envs:CartPoleEnvNoOp',
)
