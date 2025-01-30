# agent.py
import torch
import numpy as np
import itertools
import torch.optim as optim
import wandb

from dqn import DQN
# Import the new Prioritized Buffer
from replay_buffer import PrioritizedReplayBuffer
# or from replay_buffer import ReplayBuffer if not using priority

from utils import (
    EpisodeStats,
    linear_epsilon_decay,
    make_epsilon_greedy_policy,
    update_dqn
)


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
                 maxlen=100_000,
                 is_double_dqn=False,
                 use_prioritized_replay=False,
                 alpha=0.6,
                 beta_start=0.4,
                 beta_frames=200_000,
                 device="cpu"):
        """
        DQN Agent that can use Double DQN and Prioritized Replay.
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

        # For Prioritized Replay
        self.use_prioritized_replay = use_prioritized_replay
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # counts total training steps to anneal beta

        # Initialize Replay Buffer (Regular or Prioritized)
        if self.use_prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                max_size=maxlen,
                alpha=self.alpha
            )
        else:
            # If not using prioritized replay, fall back to standard replay
            from replay_buffer import ReplayBuffer
            self.buffer = ReplayBuffer(max_size=maxlen)

        # Create Q-networks
        self.q = DQN(env.observation_space.shape,
                     env.action_space.n).to(self.device)
        self.q_target = DQN(env.observation_space.shape,
                            env.action_space.n).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # Create optimizer
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        # Create epsilon-greedy policy function
        self.policy = make_epsilon_greedy_policy(self.q, env.action_space.n)

    def _beta_by_frame(self, frame):
        """
        Linearly anneal beta from beta_start -> 1 over beta_frames steps.
        """
        return min(1.0, self.beta_start + frame * (1.0 - self.beta_start) / self.beta_frames)

    def train(self, num_episodes: int = 1000):
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
        )
        current_timestep = 0

        for i_episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            for episode_time in itertools.count():
                # Update epsilon
                epsilon = linear_epsilon_decay(
                    self.eps_start, self.eps_end, current_timestep, self.schedule_duration
                )

                # Choose action (epsilon-greedy)
                action = self.policy(obs.unsqueeze(0), epsilon=epsilon)

                next_obs, reward, terminated, truncated, _ = self.env.step(
                    action)

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                next_obs_t = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=self.device)
                action_t = torch.as_tensor(action, device=self.device)
                reward_t = torch.as_tensor(
                    reward, dtype=torch.float32, device=self.device)
                done_t = torch.as_tensor(terminated, device=self.device)

                # Store transition in replay buffer
                if self.use_prioritized_replay:
                    self.buffer.store(
                        (obs, action_t, reward_t, next_obs_t, done_t))
                else:
                    self.buffer.store(obs, action_t, reward_t,
                                      next_obs_t, done_t)

                obs = next_obs_t
                current_timestep += 1
                self.frame += 1  # for beta annealing

                # Sample a batch if buffer is big enough
                if len(self.buffer) >= self.batch_size:
                    if self.use_prioritized_replay:
                        # Anneal beta
                        beta = self._beta_by_frame(self.frame)

                        (obs_b, act_b, rew_b, next_obs_b, done_b, indices, is_weights) = \
                            self.buffer.sample(self.batch_size, beta=beta)

                        # Move to device
                        obs_b = obs_b.to(self.device)
                        act_b = act_b.to(self.device)
                        rew_b = rew_b.to(self.device)
                        next_obs_b = next_obs_b.to(self.device)
                        done_b = done_b.to(self.device)
                        is_weights = is_weights.to(self.device)

                        # -- compute predicted Q and target Q for TD-error
                        self.optimizer.zero_grad()

                        # Q(s,a)
                        q_values = self.q(obs_b)
                        q_action = q_values.gather(
                            1, act_b.unsqueeze(1)).squeeze(1)

                        with torch.no_grad():
                            if self.is_double_dqn:
                                next_actions = self.q(next_obs_b).argmax(
                                    dim=1, keepdim=True)
                                q_target_next = self.q_target(next_obs_b).gather(
                                    1, next_actions).squeeze(1)
                            else:
                                q_target_next = self.q_target(
                                    next_obs_b).max(dim=1)[0]
                            td_target = rew_b + self.gamma * \
                                q_target_next * (1 - done_b.float())

                        # TD error
                        td_error = td_target - q_action

                        # Weighted MSE loss
                        loss = (is_weights * td_error**2).mean()

                        loss.backward()
                        self.optimizer.step()

                        # Update priorities in buffer
                        new_priorities = td_error.abs().detach()
                        self.buffer.update_priorities(indices, new_priorities)

                    else:
                        # Non-prioritized case
                        obs_b, act_b, rew_b, next_obs_b, done_b = self.buffer.sample(
                            self.batch_size)
                        obs_b = obs_b.to(self.device)
                        act_b = act_b.to(self.device)
                        rew_b = rew_b.to(self.device)
                        next_obs_b = next_obs_b.to(self.device)
                        done_b = done_b.to(self.device)

                        # Standard update
                        loss = update_dqn(
                            self.q,
                            self.q_target,
                            self.optimizer,
                            self.gamma,
                            obs_b,
                            act_b,
                            rew_b,
                            next_obs_b,
                            done_b,
                            is_double_dqn=self.is_double_dqn
                        )

                # Update target network
                if current_timestep % self.update_freq == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

                # End episode if done or truncated or step-limit
                if terminated or truncated or (episode_time >= 500):
                    break

            # Logging
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
