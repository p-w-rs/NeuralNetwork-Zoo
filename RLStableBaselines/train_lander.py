import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise


def cb(ldir):
    return EvalCallback(
        Monitor(env, ldir), best_model_save_path=ldir, log_path=ldir,
        eval_freq=ttime/100, n_eval_episodes=n_envs,
        deterministic=True, render=False
    )


env_id = "LunarLander-v2"
mode = "rgb_array"
ttime = 100_000
n_envs = 15

env = gym.make(
    env_id, enable_wind=True, continuous=True, render_mode=mode
)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.8 * np.ones(n_actions)
)

ddpg = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
ddpg.learn(total_timesteps=ttime, callback=cb("results/ddpg"), progress_bar=True)

td3 = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
td3.learn(total_timesteps=ttime, callback=cb("results/td3"), progress_bar=True)

sac = SAC("MlpPolicy", env, verbose=1)
sac.learn(total_timesteps=ttime, callback=cb("results/sac"), progress_bar=True)
