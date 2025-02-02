# utils.py
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def linear_epsilon_decay(eps_start: float, eps_end: float, current_timestep: int, duration: int) -> float:
    """
    Linear decay of epsilon from eps_start to eps_end over 'duration' steps.
    """
    ratio = min(1.0, current_timestep / duration)
    return (eps_start - eps_end) * (1 - ratio) + eps_end


def make_epsilon_greedy_policy(Q, num_actions: int):
    """
    Creates an epsilon-greedy policy given a Q-network and number of actions.
    """
    def policy_fn(obs: torch.Tensor, epsilon: float = 0.0):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, num_actions)
        return Q(obs).argmax(dim=1).detach().numpy()[0]  # expects batch size 1
    return policy_fn


def update_dqn(q, q_target, optimizer, gamma,
               obs, act, rew, next_obs, tm, is_double_dqn, is_distributional, num_atoms, v_min, v_max):
    """
    Update the DQN or Double DQN network for one optimizer step using the target network.
    """
    optimizer.zero_grad()

    if is_distributional:
        with torch.no_grad():
            _, next_pmfs = q_target.get_action(next_obs)
            next_atoms = rew[:, None] + gamma * q_target.support[None, :] * (1 - tm[:, None].float())
            # projection
            delta_z = q_target.support[1] - q_target.support[0]
            tz = next_atoms.clamp(v_min, v_max)

            b = (tz - v_min) / delta_z
            l = b.floor().clamp(0, num_atoms - 1)
            u = b.ceil().clamp(0, num_atoms - 1)
            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = q.get_action(obs, act.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

    else:
        if is_double_dqn:
            with torch.no_grad():
                # 1) Choose best actions from Q (online)
                next_actions = q(next_obs).argmax(
                    dim=1, keepdim=True)  # shape (batch, 1)
                # 2) Evaluate them with Q-target
                target_values = q_target(next_obs).gather(
                    1, next_actions).squeeze(1)
                td_target = rew + gamma * target_values * (1 - tm.float())
        else:
            with torch.no_grad():
            # Vanilla DQN: use max of q_target
                td_target = rew + gamma * \
                    q_target(next_obs).max(dim=1)[0] * (1 - tm.float())

        # Gather Q-values for the taken actions
        q_values = q(obs)
        q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

        # Compute MSE loss
        loss = F.mse_loss(q_action, td_target)

    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()

    return loss.item()  # for logging if desired


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
