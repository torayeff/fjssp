from gym.utils.env_checker import check_env
import gym
import fjssp

if __name__ == "__main__":
    print("Checking environment structure..")
    env = gym.make('FJSSP-v0', new_step_api=True)
    check_env(env.unwrapped)
