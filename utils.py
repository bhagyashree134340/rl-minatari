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


# def make_epsilon_greedy_policy(Q, num_actions: int):
#     """
#     Creates an epsilon-greedy policy given a Q-network and number of actions.
#     """
#     def policy_fn(obs: torch.Tensor, epsilon: float = 0.0):
#         if np.random.uniform() < epsilon:
#             return np.random.randint(0, num_actions)
#         return Q(obs).argmax(dim=1).detach().numpy()[0]  # expects batch size 1
#     return policy_fn

def make_epsilon_greedy_policy(Q, num_actions: int):
    def policy_fn(obs: torch.Tensor, epsilon: float = 0.0):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, num_actions)

        obs = obs.unsqueeze(0) if len(obs.shape) == 3 else obs  # ✅ Ensure batch dim
        return Q(obs).argmax(dim=1).detach().cpu().numpy()[0]  # ✅ Convert to numpy
    return policy_fn




def update_dqn(q, q_target, optimizer, gamma,
               obs, act, rew, next_obs, tm, is_double_dqn=False):
    """
    Update the DQN or Double DQN network for one optimizer step using the target network.
    """
    optimizer.zero_grad()

    with torch.no_grad():
        if is_double_dqn:
            # Double DQN: Select action using q (online) & evaluate using q_target
            next_actions = q(next_obs).argmax(dim=1, keepdim=True)  
            target_values = q_target(next_obs).gather(1, next_actions).squeeze(1)
            td_target = rew + gamma * target_values * (1 - tm.float())
        else:
            # Vanilla DQN: Use max Q-value from target network
            td_target = rew + gamma * q_target(next_obs).max(dim=1)[0] * (1 - tm.float())

    # Compute current Q-values for chosen actions
    q_values = q(obs)
    q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

    # Compute MSE loss and optimize
    loss = F.mse_loss(q_action, td_target)
    loss.backward()
    optimizer.step()

    return loss.item()


def update_dual_dqn(q1, q2, q_target1, q_target2, optimizer1, optimizer2, gamma, 
                     obs, act, rew, next_obs, tm):
    """
    Update both networks in Dual DQN.

    - Uses two Q-networks and two target networks.
    - Updates each using the minimum Q-value approach for stability.
    """
    optimizer1.zero_grad()
    optimizer2.zero_grad()

    with torch.no_grad():
        # Compute target using min(Q1', Q2') to reduce overestimation bias
        q1_next = q_target1(next_obs).max(dim=1)[0]
        q2_next = q_target2(next_obs).max(dim=1)[0]
        min_q_next = torch.min(q1_next, q2_next)

        td_target = rew + gamma * min_q_next * (1 - tm.float())

    # Compute Q-value estimates
    q1_pred = q1(obs).gather(1, act.unsqueeze(1)).squeeze(1)
    q2_pred = q2(obs).gather(1, act.unsqueeze(1)).squeeze(1)

    # Compute loss for each network
    loss1 = F.mse_loss(q1_pred, td_target)
    loss2 = F.mse_loss(q2_pred, td_target)

    # Optimize both networks
    loss1.backward()
    optimizer1.step()

    loss2.backward()
    optimizer2.step()

    return loss1.item(), loss2.item()


def update_multi_step(q, q_target, optimizer, gamma, obs, act, rew, next_obs, tm, n_step, gamma_n):
    """
    Update function for Multi-Step Learning DQN.

    - Uses N-step returns for training stability.
    - Helps propagate rewards faster in temporal learning.
    """
    optimizer.zero_grad()

    with torch.no_grad():
        # Compute the n-step return target
        next_q_values = q_target(next_obs).max(dim=1)[0]
        td_target = rew + gamma_n * next_q_values * (1 - tm.float())

    # Compute current Q-values for chosen actions
    q_values = q(obs)
    q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

    # Compute MSE loss and optimize
    loss = F.mse_loss(q_action, td_target)
    loss.backward()
    optimizer.step()

    return loss.item()


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
