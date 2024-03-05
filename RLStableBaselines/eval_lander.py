import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

env_id = "LunarLander-v2"
mode = "rgb_array"
ttime = 100_000

env = gym.make(
    env_id, enable_wind=True, continuous=True, render_mode=mode
)

ddpg = DDPG.load("results/ddpg/best_model.zip", env)
td3 = TD3.load("results/td3/best_model.zip", env)
sac = SAC.load("results/sac/best_model.zip", env)

for (model, ldir) in zip([ddpg, td3, sac], ["results/ddpg", "results/td3", "results/sac"]):
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=15
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    plot_results([ldir], ttime, results_plotter.X_TIMESTEPS, "LunarLander-v2")
    plt.show()
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
