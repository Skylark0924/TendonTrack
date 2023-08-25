from gym.envs.registration import register

register(
    id='tendontrack-v0',
    entry_point='gym_tendontrack.envs:TendonTrackEnv',
)
