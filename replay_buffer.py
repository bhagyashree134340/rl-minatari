import random
import torch
import numpy as np
from utils import set_seed
set_seed(42)

class ReplayBuffer:
    def __init__(self, max_size: int):
        """
        Create the replay buffer.

        :param max_size: Maximum number of transitions in the buffer.
        """
        self.data = []
        self.max_size = max_size
        self.position = 0

    def __len__(self) -> int:
        return len(self.data)

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
              next_obs: torch.Tensor, terminated: torch.Tensor):
        """
        Adds a new transition to the buffer. When the buffer is full, overwrite the oldest transition.
        """
        if len(self.data) < self.max_size:
            self.data.append((obs, action, reward, next_obs, terminated))
        else:
            self.data[self.position] = (
                obs, action, reward, next_obs, terminated)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions (with replacement).
        Returns a tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch).
        """
        batch = random.choices(self.data, k=batch_size)
        # transpose the list of tuples
        obs, act, rew, next_obs, ter = zip(*batch)

        # stack each
        return (torch.stack(obs),
                torch.stack(act),
                torch.stack(rew),
                torch.stack(next_obs),
                torch.stack(ter))
    
class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, epsilon=1e-5):
        """
        Proportional Prioritized Replay Buffer
        :param max_size: Maximum number of transitions
        :param alpha: how much prioritization is used (0 - no prioritization, 1 - full)
        :param epsilon: small constant to avoid zero priority
        """
        self.max_size = max_size
        self.alpha = alpha
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def store(self, transition):
        """
        Stores a transition.
        transition should be a tuple: (obs, action, reward, next_obs, done)
        """
        # If new transition is appended for the first time, default priority = max priority or 1.0
        # (Since the new transition has no TD error yet, we'll guess it might be important.)
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
            # Use the max priority so that new samples get a chance to be visited
            if len(self.priorities) > 0:
                max_prio = max(self.priorities)
            else:
                max_prio = 1.0
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            # overwrite priority
            if len(self.priorities) > 0:
                max_prio = max(self.priorities)
            else:
                max_prio = 1.0
            self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions, returning both data and importance-sampling (IS) weights.
        :param beta: importance-sampling exponent to compensate for non-uniform p.
        :returns: (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, indices, is_weights)
        """
        if len(self.buffer) == self.max_size:
            # Use all priorities
            prios = np.array(self.priorities)
        else:
            prios = np.array(self.priorities[: len(self.buffer)])

        # Compute probabilities using priorities^alpha
        scaled_prios = prios ** self.alpha
        sample_probs = scaled_prios / scaled_prios.sum()

        # Sample according to these probabilities
        indices = np.random.choice(
            len(self.buffer), batch_size, p=sample_probs)

        # Compute importance-sampling weights
        # w_i = (1/(N * P(i)))^beta
        total = len(self.buffer)
        weights = (total * sample_probs[indices]) ** (-beta)
        # Normalize weights by max for stability
        weights /= weights.max()

        # Fetch transitions
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = [], [], [], [], []
        for i in indices:
            o, a, r, no, d = self.buffer[i]
            obs_batch.append(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))   # keep them as Tensors
            act_batch.append(a.unsqueeze(0))
            rew_batch.append(r.unsqueeze(0))
            next_obs_batch.append(no.unsqueeze(0))
            done_batch.append(d.unsqueeze(0))

        # Concatenate
        obs_batch = torch.cat(obs_batch, dim=0)
        act_batch = torch.cat(act_batch, dim=0)
        rew_batch = torch.cat(rew_batch, dim=0)
        next_obs_batch = torch.cat(next_obs_batch, dim=0)
        done_batch = torch.cat(done_batch, dim=0)

        # Convert weights to torch
        is_weights = torch.as_tensor(weights, dtype=torch.float32)

        return (obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, indices, is_weights)

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled transitions given new TD-errors.
        :param indices: the indices in the buffer of these samples
        :param priorities: the new priorities (absolute TD error)
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio.item() + self.epsilon