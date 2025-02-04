# dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_linear import NoisyLinear
from utils import set_seed
set_seed(42)


class DQN(nn.Module):
    def __init__(self, obs_shape, num_actions, is_noisy_nets=False, std_init = 0.4, is_distributional=False, num_atoms=51, v_min=-10, v_max=10):
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
        self.std_init = std_init
        self.is_distributional = is_distributional
        self.num_actions = num_actions
        self.num_atoms = num_atoms if is_distributional else 1  # Default to 1 if not distributional
        self.v_min = v_min
        self.v_max = v_max
        # self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=num_atoms))

        # Define the network architecture
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, stride=1, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, stride=1, kernel_size=3)

        # Fully connected layers
        if self.is_noisy_nets:
            self.fc1 = NoisyLinear(32 * 4 * 4, 128, std_init=std_init)
            self.fc2 = NoisyLinear(128, num_actions * self.num_atoms, std_init=std_init)
        else:
            self.fc1 = nn.Linear(32 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, num_actions * self.num_atoms)

        # Support for Distributional RL
        if self.is_distributional:
            self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Switch from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = torch.flatten(x, 1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        # if self.is_distributional:
        #     x = x.view(-1, self.num_actions, self.num_atoms)
        #     return F.softmax(x, dim=-1)  # Return probabilities over atoms
        # else:
        #     # Standard Q-values
        #     return x
        return x

    def reset_noise(self):
        if self.is_noisy_nets:
            for layer in [self.fc1, self.fc2]:
                layer.reset_noise()

    def get_action(self, x, action=None):
        logits = self.forward(x / 255.0)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.num_actions, self.num_atoms), dim=2)
        q_values = (pmfs * self.support).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]