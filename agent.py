

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # ✅ Ensure correct imports

import torch
import numpy as np
import itertools
import torch.optim as optim
import wandb

from dqn import DQN
from dueling_networks import DuelingDQNAgent, update_dueling_dqn  
from replay_buffer import PrioritizedReplayBuffer  
from multi_step_learning import MultiStepBuffer, update_multi_step_dqn  # ✅ Added Multi-Step Learning
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
                 is_double_dqn= False,
                 is_dueling_dqn= False,  
                 is_noisy_nets= False,
                 std_init=0.4,
                 is_distributional= False,
                 num_atoms=51,
                 v_min=-10,
                 v_max=10,
                 maxlen=100_000,
                 use_prioritized_replay= False,
                 use_multi_step= False,  
                 n_steps=3,  
                 alpha=0.6,
                 beta_start=0.4,
                 beta_frames=200_000,
                 device="cpu"):
        """
        DQN Agent that supports:
        - Double DQN
        - Dueling Networks
        - Noisy Nets
        - Distributional RL
        - Prioritized Replay Buffer
        - Multi-Step Learning
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
        self.is_dueling_dqn = is_dueling_dqn
        self.is_noisy_nets = is_noisy_nets
        self.std_init = std_init
        self.is_distributional = is_distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_prioritized_replay = use_prioritized_replay  
        self.use_multi_step = use_multi_step  
        self.n_steps = n_steps  
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # ✅ Training step counter for beta annealing

        # Initialize Replay Buffer (Multi-Step > Prioritized > Regular)
        if self.use_multi_step:
            self.buffer = MultiStepBuffer(
                maxlen=maxlen,
                n_steps=self.n_steps,
                gamma=self.gamma,
                device=self.device
            )
        elif self.use_prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                max_size=maxlen,
                alpha=self.alpha
            )
        else:
            from replay_buffer import ReplayBuffer
            self.buffer = ReplayBuffer(max_size=maxlen)

        # Initialize Networks
        if self.is_dueling_dqn:
            self.dueling_agent = DuelingDQNAgent(
                env, lr, device, is_noisy_nets, std_init, is_distributional, num_atoms, v_min, v_max
            )  
        else:
            self.q = DQN(env.observation_space.shape, env.action_space.n, is_noisy_nets, std_init, is_distributional, num_atoms, v_min, v_max).to(self.device)
            self.q_target = DQN(env.observation_space.shape, env.action_space.n, is_noisy_nets, std_init, is_distributional, num_atoms, v_min, v_max).to(self.device)

            self.q_target.load_state_dict(self.q.state_dict())
            self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        # Create epsilon-greedy policy function
        if not self.is_noisy_nets:
            self.policy = make_epsilon_greedy_policy(
                self.q if not self.is_dueling_dqn else self.dueling_agent.q, env.action_space.n
            )

    def _beta_by_frame(self, frame):
        """Linearly anneal beta from beta_start -> 1 over beta_frames steps."""
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

            # Reset Noisy Nets if needed
            if self.is_noisy_nets:
                if self.is_dueling_dqn:
                    self.dueling_agent.q.reset_noise()
                else:
                    self.q.reset_noise()

            for episode_time in itertools.count():
                # Action Selection: Noisy Nets or Epsilon-Greedy
                if self.is_noisy_nets:
                    action = self.q(obs.unsqueeze(0)).argmax(dim=1).item() if not self.is_dueling_dqn else self.dueling_agent.q(obs.unsqueeze(0)).argmax(dim=1).item()
                else:
                    epsilon = linear_epsilon_decay(self.eps_start, self.eps_end, current_timestep, self.schedule_duration)
                    if self.is_distributional:
                        if np.random.uniform() < epsilon:
                            action = np.random.randint(0, self.env.action_space.n)
                        else:
                            #FIXME: pmf 
                            action, pmf = self.q.get_action(obs.unsqueeze(0))
                            action = action.item()
                    else:
                        action = self.policy(obs.unsqueeze(0), epsilon=epsilon)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

                #  Store in replay buffer (Multi-Step, Prioritized, or Regular)
                transition = (obs,  
                            torch.as_tensor(action, device=self.device),  
                            torch.as_tensor(reward, dtype=torch.float32, device=self.device),  
                            next_obs,  
                            torch.as_tensor(terminated, device=self.device))

                if self.use_multi_step:
                    self.buffer.store(*transition)  #  Multi-Step Buffer
                elif self.use_prioritized_replay:
                    self.buffer.store(transition)  # Prioritized Replay
                else:
                    self.buffer.store(*transition)  # Regular Replay

                obs = next_obs
                current_timestep += 1
                self.frame += 1  

                # Ensure buffer has enough samples before sampling
                if len(self.buffer) >= self.batch_size:
                    if self.use_prioritized_replay:
                        beta = self._beta_by_frame(self.frame)
                        batch = self.buffer.sample(self.batch_size, beta=beta)
                        obs_b, act_b, rew_b, next_obs_b, done_b, indices, is_weights = batch
                    elif self.use_multi_step:
                        batch = self.buffer.sample(self.batch_size)
                        obs_b, act_b, rew_b, next_obs_b, done_b = batch
                    else:
                        batch = self.buffer.sample(self.batch_size)
                        obs_b, act_b, rew_b, next_obs_b, done_b = batch

                    #  Ensure sampled variables are correctly assigned before use
                    if obs_b is not None:
                        if self.is_dueling_dqn:
                            update_dueling_dqn(self.dueling_agent, self.gamma, obs_b, act_b, rew_b, next_obs_b, done_b, is_double_dqn=self.is_double_dqn)
                        elif self.use_multi_step:
                            update_multi_step_dqn(
                                self.q, self.q_target, self.optimizer, self.gamma, self.n_steps,
                                obs_b, act_b, rew_b, next_obs_b, done_b
                            )
                        else:
                            update_dqn(
                                self.q, 
                                self.q_target, 
                                self.optimizer, 
                                self.gamma, 
                                obs_b, act_b, rew_b, next_obs_b, done_b,
                                is_double_dqn=self.is_double_dqn,
                                is_distributional=self.is_distributional,  
                                is_dueling_dqn=self.is_dueling_dqn,  
                                num_atoms=self.num_atoms, 
                                v_min=self.v_min, 
                                v_max=self.v_max
                            )

                #  Update target network periodically
                if current_timestep % self.update_freq == 0:
                    if self.is_dueling_dqn:
                        self.dueling_agent.update_target_network()
                    else:
                        self.q_target.load_state_dict(self.q.state_dict())

                #  Terminate episode if done
                if terminated or truncated or episode_time >= 500:
                    break

            # Logging progress
            wandb.log({
                "episode": i_episode,
                "epsilon": epsilon if not self.is_noisy_nets else 'Noisy Nets',
                "episode_reward": stats.episode_rewards[i_episode],
                "episode_length": stats.episode_lengths[i_episode],
            })

        return stats

