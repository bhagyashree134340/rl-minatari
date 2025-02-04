# agent.py
import torch
import numpy as np
import itertools
from dqn import DQN
from replay_buffer import ReplayBuffer
from utils import (EpisodeStats,
                   linear_epsilon_decay,
                   make_epsilon_greedy_policy,
                   update_dqn)

import torch.optim as optim
import wandb
from utils import set_seed
set_seed(42)


class DQNAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 lr=0.001,
                 batch_size=64,
                 eps_start=1.0,
                 eps_end=0.1,
                 schedule_duration=10_000,
                 update_freq=100,
                 is_double_dqn=False,
                 is_noisy_nets=False,
                 std_init = 0.4,
                 is_distributional=False,
                 num_atoms=51, 
                 v_min=-10, 
                 v_max=10,
                 maxlen=100_000,
                 device="cpu"):
        """
        Initialize the DQN agent.
        """
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_duration = schedule_duration
        self.update_freq = update_freq
        self.device = device
        self.is_double_dqn = is_double_dqn
        self.is_noisy_nets = is_noisy_nets
        self.std_init = std_init
        self.is_distributional = is_distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Initialize Replay Buffer
        self.buffer = ReplayBuffer(maxlen)

        # Create Q-networks
        self.q = DQN(env.observation_space.shape, env.action_space.n, is_noisy_nets, std_init, is_distributional, num_atoms, v_min, v_max).to(self.device)
        self.q_target = DQN(env.observation_space.shape, env.action_space.n, is_noisy_nets, std_init, is_distributional, num_atoms, v_min, v_max).to(self.device)

        self.q_target.load_state_dict(self.q.state_dict())

        # Create optimizer
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        # Create epsilon-greedy policy function
        if not self.is_noisy_nets:
            self.policy = make_epsilon_greedy_policy(self.q, env.action_space.n)

    def train(self, num_episodes: int = 1000):
        """
        Train the DQN agent.
        """
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
        )
        current_timestep = 0

        for i_episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            if self.is_noisy_nets:
                self.q.reset_noise()

            for episode_time in itertools.count():
                #if noisy nets then no epsilon-greedy
                if self.is_noisy_nets:
                    action = self.q(obs.unsqueeze(0)).argmax(dim=1).item()
                else:
                    epsilon = linear_epsilon_decay(
                        self.eps_start, self.eps_end, current_timestep, self.schedule_duration)

                    if self.is_distributional:
                        if np.random.uniform() < epsilon:
                            action = np.random.randint(0, self.env.action_space.n)
                        else:
                            #FIXME: pmf doesnt reach update
                            action, pmf = self.q.get_action(obs.unsqueeze(0))
                            action = action.item()
                    else:
                        action = self.policy(obs.unsqueeze(0), epsilon=epsilon)

                # print(action)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                next_obs = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=self.device)

                # Store in replay buffer
                self.buffer.store(obs,
                                  torch.as_tensor(action, device=self.device),
                                  torch.as_tensor(
                                      reward, dtype=torch.float32, device=self.device),
                                  next_obs,
                                  torch.as_tensor(terminated, device=self.device))

                # Sample a batch if buffer is big enough
                if len(self.buffer) >= self.batch_size:
                    obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = self.buffer.sample(
                        self.batch_size)
                    # move to device if needed
                    obs_batch = obs_batch.to(self.device)
                    act_batch = act_batch.to(self.device)
                    rew_batch = rew_batch.to(self.device)
                    next_obs_batch = next_obs_batch.to(self.device)
                    tm_batch = tm_batch.to(self.device)

                    # Update DQN
                    loss = update_dqn(
                        self.q,
                        self.q_target,
                        self.optimizer,
                        self.gamma,
                        obs_batch,
                        act_batch,
                        rew_batch,
                        next_obs_batch,
                        tm_batch,
                        self.is_double_dqn,
                        self.is_distributional, 
                        self.num_atoms, 
                        self.v_min, 
                        self.v_max
                    )

                # Update target network
                if current_timestep % self.update_freq == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

                current_timestep += 1

                # End episode if done or we hit a step limit
                if terminated or truncated or episode_time >= 500:
                    break

                obs = next_obs

            # Print progress
            if self.is_noisy_nets:
                if (i_episode + 1) % 100 == 0:
                    print(f"Episode {i_episode+1}/{num_episodes} "
                        f"Time Step: {current_timestep} "
                        f"Reward: {stats.episode_rewards[i_episode]:.2f}")

                wandb.log({
                    "episode": i_episode,
                    "episode_reward": stats.episode_rewards[i_episode],
                    "episode_length": stats.episode_lengths[i_episode],
                })
            else:
                if (i_episode + 1) % 100 == 0:
                    print(f"Episode {i_episode+1}/{num_episodes} "
                        f"Time Step: {current_timestep} "
                        f"Epsilon: {epsilon:.3f} "
                        f"Reward: {stats.episode_rewards[i_episode]:.2f}")

                wandb.log({
                    "episode": i_episode,
                    "epsilon": epsilon,
                    "episode_reward": stats.episode_rewards[i_episode],
                    "episode_length": stats.episode_lengths[i_episode],
                })
        return stats
