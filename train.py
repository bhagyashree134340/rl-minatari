import wandb
import gymnasium as gym
import numpy as np
import torch
from agent import DQNAgent
import matplotlib.pyplot as plt
from utils import make_epsilon_greedy_policy
import pandas as pd
from IPython.display import Image as IImage
import yaml
from utils import set_seed
set_seed(42)


def main(is_noisy_nets=False):
    wandb.init(
        project="minatar-dqn",
        name="DQN-Breakout-Noisy" if is_noisy_nets else "DQN-Breakout",
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
            "is_noisy_nets": is_noisy_nets, 
        }
    )
    config = wandb.config

    env = gym.make('MinAtar/Breakout-v1', render_mode="rgb_array")
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
        is_double_dqn=False,  # double DQN is set to TRUE
        is_noisy_nets=config.is_noisy_nets,
        is_distributional=True,
        num_atoms=21, 
        v_min=-10, 
        v_max=10,
    )

    stats = agent.train(config.num_episodes)

    # Log final results, e.g. average reward of last 50 episodes
    avg_reward = np.mean(stats.episode_rewards[-50:])
    wandb.log({"final_average_reward": avg_reward})
    print(f"Final Average Reward (last 50 episodes): {avg_reward}")

    def plot_and_log(stats, smoothing_window=10):
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

        # 1) Episode Lengths
        ax = axes[0]
        ax.plot(stats.episode_lengths)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length")
        ax.set_title("Episode Length over Time")

        # 2) Smoothed Episode Rewards
        ax = axes[1]
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
            smoothing_window, min_periods=smoothing_window
        ).mean()
        ax.plot(rewards_smoothed)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.set_title(
            f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})"
        )

        # Log this figure to W&B
        wandb.log({"training_plots": wandb.Image(fig)})

        # Close the figure (to avoid memory issues in notebook or repeated logging)
        plt.close(fig)

    plot_and_log(stats=stats, smoothing_window=20)


# def main():

#     with open("./sweep_config.yaml") as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)
#     # 1. Initialize W&B run
#     print(config)
#     wandb.init(project="my-minatar-project", config=config)

#     # 2. Set seeds
#     set_seed(42)

#     # 3. Create environment
#     env = gym.make('MinAtar/Breakout-v1', render_mode="rgb_array")

#     # 4. Create agent
#     agent = DQNAgent(
#         env=env,
#         gamma=0.99,
#         lr=wandb.config.lr,                  # from sweep
#         batch_size=wandb.config.batch_size,  # from sweep
#         eps_start=0.5,
#         eps_end=wandb.config.eps_end,        # from sweep
#         schedule_duration=15_000,
#         update_freq=100,
#         maxlen=100_000,
#         # is_double_dqn=wandb.config.is_double_dqn,  # toggle Double vs. Vanilla
#         device="cpu"
#     )

#     # 5. Train
#     num_episodes = wandb.config.episode
#     stats = agent.train(num_episodes)

#     # 6. Compute some metric to report back to W&B
#     avg_reward = np.mean(stats.episode_rewards[-50:])  # last 50 episodes
#     wandb.log({"avg_reward": avg_reward})

#     print(f"Finished run:, "
#           f"LR={wandb.config.lr}, Batch={wandb.config.batch_size}, eps_end={wandb.config.eps_end}, "
#           f"AvgReward={avg_reward:.2f}")
if __name__ == "__main__":
    main(is_noisy_nets=False)
