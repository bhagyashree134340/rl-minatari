# dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_linear import NoisyLinear
from utils import set_seed
set_seed(42)


class DQN(nn.Module):
    def __init__(self, obs_shape, num_actions, is_noisy_nets=False, is_distributional=False):
        """
        Initialize the DQN network.

        :param obs_shape: Shape of the observation space
        :param num_actions: Number of actions
        :param is_noisy_nets: Whether to use NoisyNets
        :param is_distributional: Whether to use Distributional RL
        :param num_atoms: Number of atoms for Distributional RL
        :param v_min: Minimum value of the support for Distributional RL
        :param v_max: Maximum value of the support for Distributional RL
        """
        super(DQN, self).__init__()

        self.is_noisy_nets = is_noisy_nets
        self.is_distributional = is_distributional

        self.num_actions = num_actions
        self.num_atoms = 51 if is_distributional else 1  # Default to 1 if not distributional
        self.v_min = -10
        self.v_max = 10
        # self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to("cpu")

        # Define the network architecture
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, stride=1, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, stride=1, kernel_size=3)

        # Fully connected layers
        if self.is_noisy_nets:
            self.fc1 = NoisyLinear(32 * 4 * 4, 128)
            self.fc2 = NoisyLinear(128, num_actions * self.num_atoms)
        else:
            self.fc1 = nn.Linear(32 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, num_actions * self.num_atoms)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Switch from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = torch.flatten(x, 1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        if self.is_distributional:
            x = x.view(-1, self.num_actions, self.num_atoms)
            return F.softmax(x, dim=-1)  # Return probabilities over atoms
        else:
            # Standard Q-values
            return x

    def reset_noise(self):
        if self.is_noisy_nets:
            for layer in [self.fc1, self.fc2]:
                layer.reset_noise()