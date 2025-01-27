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
    def policy_fn(obs: torch.Tensor, epsilon: float = 0.0, q_values: torch.Tensor = None):
        if q_values is None:
            q_values = Q(obs)
        else:
            q_values = q_values

        if np.random.uniform() < epsilon:
            return np.random.randint(0, num_actions)
        return q_values.argmax(dim=1).detach().numpy()[0]  # expects batch size 1
    return policy_fn


def update_dqn(q, q_target, optimizer, gamma,
               obs, act, rew, next_obs, tm, is_double_dqn=False, is_distributional=False):
    """
    Update the DQN or Double DQN network for one optimizer step using the target network.
    """
    optimizer.zero_grad()

    if is_distributional:
        with torch.no_grad():
            # Compute target distribution
            next_probs = q_target(next_obs)  # (B, A, num_atoms)
            next_actions = next_probs.mean(dim=2).argmax(dim=1)  # (B,)
            next_probs = next_probs[torch.arange(next_probs.size(0)), next_actions]  # (B, num_atoms)

            # Project target distribution onto support
            delta_z = (q.v_max - q.v_min) / (q.num_atoms - 1)
            target_support = rew.unsqueeze(1) + gamma * q.support * (1 - tm.float()).unsqueeze(1)
            target_support = torch.clamp(target_support, q.v_min, q.v_max)

            # Compute projection of target_support onto support
            b = (target_support - q.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Distribute probabilities
            proj_probs = torch.zeros_like(next_probs)
            proj_probs.scatter_add_(1, l, next_probs * (u.float() - b))
            proj_probs.scatter_add_(1, u, next_probs * (b - l.float()))

        # Compute predicted distribution
        logits = q(obs)  # (B, A, num_atoms)
        logits = logits[torch.arange(logits.size(0)), act]  # (B, num_atoms)

        # Compute cross-entropy loss
        loss = -(proj_probs * logits.log_softmax(dim=1)).sum(dim=1).mean()
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
