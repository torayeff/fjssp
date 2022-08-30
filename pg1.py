import sys
sys.path.insert(1, "../../")
import gym
import fjssp

from stable_baselines3 import PPO

env = gym.make("FJSSP-v0", new_step_api=False)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    if done:
      obs = env.reset()

env.close()