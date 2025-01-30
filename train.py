import wandb
import gymnasium as gym
import numpy as np
import torch
import yaml
from agent import DQNAgent
import matplotlib.pyplot as plt
import pandas as pd
from utils import make_epsilon_greedy_policy, set_seed


def main():
    wandb.init(
        project="minatar-dqn",
        name="DQN-Breakout",
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
        }
    )
    config = wandb.config
    set_seed(42)

    env = gym.make('MinAtar/Breakout-v1', render_mode="rgb_array")

    agent = DQNAgent(
        env=env,
        gamma=config.discount_factor,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        eps_start=config.eps_start,
        eps_end=config.eps_end,
        schedule_duration=config.schedule_duration,
        update_freq=config.update_freq,
        maxlen=config.replay_buffer_size,
        is_double_dqn=config.is_double_dqn,
        use_prioritized_replay=config.use_prioritized_replay,
        device="cpu"
    )

    stats = agent.train(config.num_episodes)

    # Log final results
    avg_reward = np.mean(stats.episode_rewards[-50:])
    wandb.log({"final_average_reward": avg_reward})
    print(f"Final Average Reward (last 50 episodes): {avg_reward}")

    # Quick plotting function
    def plot_and_log(stats, smoothing_window=10):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

        ax = axes[0]
        ax.plot(stats.episode_lengths)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length")
        ax.set_title("Episode Length over Time")

        ax = axes[1]
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
            smoothing_window, min_periods=smoothing_window
        ).mean()
        ax.plot(rewards_smoothed)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.set_title(
            f"Episode Reward over Time (Smoothed over window size {smoothing_window})"
        )

        wandb.log({"training_plots": wandb.Image(fig)})
        plt.close(fig)

    plot_and_log(stats, smoothing_window=20)


if __name__ == "__main__":
    main()
