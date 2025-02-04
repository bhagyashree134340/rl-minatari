

import wandb
import gymnasium as gym
import numpy as np
from agent import DQNAgent
import matplotlib.pyplot as plt
from utils import animate
import pandas as pd
from utils import animate, set_seed
import yaml
from utils import set_seed
from minatar import Environment

# ✅ Set seed for reproducibility
set_seed(42)

# ✅ Custom MinAtar Wrapper
class MinAtarGymWrapper(gym.Env):
    """Custom Gym Wrapper for MinAtar environments."""

    def __init__(self, game="breakout"):
        super().__init__()
        self.env = Environment(game)
        
        # ✅ Properly set action & observation spaces
        self.action_space = gym.spaces.Discrete(self.env.num_actions())  
        obs_shape = self.env.state().shape  
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.env.reset()
        return self.env.state(), {}  

    def step(self, action):
        reward, done = self.env.act(action)  
        return self.env.state(), reward, done, False, {}  

    def render(self, mode="rgb_array"):
        return self.env.state()  

    def close(self):
        pass


def main():
    wandb.init(
        project="minatar-dqn",
        name="DQN-MinAtar",
        config={
            "learning_rate": 0.001,
            "batch_size": 8,
            "replay_buffer_size": 100_000,
            "update_freq": 100,
            "eps_start": 0.5,
            "eps_end": 0.05,
            "schedule_duration": 15_000,
            "num_episodes": 1000,
            "discount_factor": 0.99,
            "is_double_dqn": True,
            "use_prioritized_replay": True,  # Toggle this to see difference 
            "is_noisy_nets": True, 
            "std_init":0.5,
            "is_distributional":False,
            "num_atoms":51,
            "is_dueling_dqn": True,  # ✅ Fixed: Changed from `is_dual_dqn`
            "is_multi_step": False,  
            "multi_step_n": 3,  
        }                   
    )
    config = wandb.config

    # ✅ Use our MinAtar wrapper
    env = MinAtarGymWrapper("breakout")
    print(f"✅ Successfully created MinAtar Environment: {env}")
    obs, _ = env.reset()
    print(f"DEBUG: Initial Observation Shape: {obs.shape}")  

    agent = DQNAgent(
        env,
        gamma=config.discount_factor,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        eps_start=config.eps_start,
        eps_end=config.eps_end,
        schedule_duration=config.schedule_duration,
        update_freq=config.update_freq,
        maxlen=config.replay_buffer_size,
        is_double_dqn=config.is_double_dqn,  # double DQN is set to TRUE
        is_noisy_nets=config.is_noisy_nets,
        std_init=config.std_init,
        is_distributional=config.is_distributional,
        num_atoms=config.num_atoms, 
        v_min=-10, 
        v_max=10,
        is_dueling_dqn=config.is_dueling_dqn,  # ✅ Fixed: Changed from `is_dual_dqn`
        is_multi_step=config.is_multi_step,
        multi_step_n=config.multi_step_n if config.is_multi_step else 1,
    )

    stats = agent.train(config.num_episodes)

    # ✅ Log final results
    avg_reward = np.mean(stats.episode_rewards[-50:])
    wandb.log({"final_average_reward": avg_reward})
    print(f"✅ Final Average Reward (last 50 episodes): {avg_reward}")

    def plot_and_log(stats, smoothing_window=10):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

        # ✅ Episode Lengths
        ax = axes[0]
        ax.plot(stats.episode_lengths)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length")
        ax.set_title("Episode Length over Time")

        # ✅ Smoothed Episode Rewards
        ax = axes[1]
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        ax.plot(rewards_smoothed)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.set_title(f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})")

        # ✅ Log to W&B
        wandb.log({"training_plots": wandb.Image(fig)})
        plt.close(fig)

    plot_and_log(stats=stats, smoothing_window=20)
    animate(env, agent, agent.is_noisy_nets, agent.is_distributional, agent.is_double_dqn)

if __name__ == "__main__":
    main()
