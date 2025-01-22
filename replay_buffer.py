# replay_buffer.py
import random
import torch
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
