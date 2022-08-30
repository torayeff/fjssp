from gym.envs.registration import register

register(
    id="FJSSP-v0",
    entry_point="fjssp.envs:FJSSPEnv",
)