from gym.envs.registration import register

register(
    id='counterfac-v0',
    entry_point='gym_counterfac.envs:CounterFac',
)
