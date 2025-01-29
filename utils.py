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


def make_epsilon_greedy_policy(Q, num_actions: int, is_distributional:bool, num_atoms:int, v_min:int, v_max:int):
    """
    Creates an epsilon-greedy policy given a Q-network and number of actions.
    """
    def policy_fn(obs: torch.Tensor, epsilon: float = 0.0):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, num_actions)
        with torch.no_grad():
            q_values = Q(obs)  # Shape: (batch_size, num_actions, num_atoms) if distributional

            if is_distributional:
                # Compute expected Q-values for each action
                support = torch.linspace(v_min, v_max, num_atoms, device=obs.device)
                expected_q_values = (q_values * support).sum(dim=-1) 
                action = expected_q_values.argmax(dim=1).item() 
            else:
                action = q_values.argmax(dim=1).item() 
        return action 
    return policy_fn



def update_dqn(q, q_target, optimizer, gamma,
               obs, act, rew, next_obs, tm, is_double_dqn=False, is_distributional=False):
    """
    Update the DQN or Double DQN network for one optimizer step using the target network.
    """
    optimizer.zero_grad()

    if is_distributional:
        with torch.no_grad():
            # # Compute target distribution
            # next_probs = q_target(next_obs)  # (B, A, num_atoms)
            # next_actions = next_probs.mean(dim=2).argmax(dim=1)  # (B,)
            # next_probs = next_probs[torch.arange(next_probs.size(0)), next_actions]  # (B, num_atoms)

            # # Project target distribution onto support
            delta_z = (q.v_max - q.v_min) / (q.num_atoms - 1)
            support = torch.linspace(q.v_min, q.v_max, q.num_atoms, device=obs.device)
            # target_support = rew.unsqueeze(1) + gamma * q.support * (1 - tm.float()).unsqueeze(1)
            # target_support = torch.clamp(target_support, q.v_min, q.v_max)

            next_dist = q_target(next_obs)
            next_actions = next_dist.mean(2).argmax(1)
            next_dist = next_dist[range(len(next_actions)), next_actions]

            target_support = rew.unsqueeze(1) + gamma * support.unsqueeze(0) * (1 - tm.float()).unsqueeze(1)
            target_support = torch.clamp(target_support, q.v_min, q.v_max)

            # Compute projection of target_support onto support
            b = (target_support - q.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (len(rew) - 1) * q.num_atoms, len(rew)).long().unsqueeze(1).expand(len(rew), q.num_atoms).to(obs.device)
            
            proj_dist = torch.zeros_like(next_dist)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

         # Compute the current distribution
        current_dist = q(obs)
        actions = act.unsqueeze(1).expand(len(act), q.num_atoms)
        current_dist = current_dist.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the cross-entropy loss
        loss = -(proj_dist * torch.log(current_dist + 1e-8)).sum(1).mean()
    else:
        # TD Target or double dqn
        with torch.no_grad():
            if is_double_dqn:
                # 1) Choose best actions from Q (online)
                next_actions = q(next_obs).argmax(
                    dim=1, keepdim=True)  # shape (batch, 1)
                # 2) Evaluate them with Q-target
                target_values = q_target(next_obs).gather(
                    1, next_actions).squeeze(1)
                td_target = rew + gamma * target_values * (1 - tm.float())
            else:
                # Vanilla DQN: use max of q_target
                td_target = rew + gamma * \
                    q_target(next_obs).max(dim=1)[0] * (1 - tm.float())

        # Gather Q-values for the taken actions
        q_values = q(obs)
        q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

        # Compute MSE loss
        loss = F.mse_loss(q_action, td_target)

    loss.backward()
    optimizer.step()

    return loss.item()  # for logging if desired


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
