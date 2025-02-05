import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from noisy_linear import NoisyLinear  # ✅ Import Noisy Linear for Noisy Nets


class DuelingNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, is_noisy_nets=False, std_init=0.4, is_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        """
        Implements Dueling Q-Network.
        :param obs_shape: Observation space shape
        :param num_actions: Number of actions
        :param is_noisy_nets: Whether to use NoisyNets
        :param is_distributional: Whether to use Distributional RL
        :param num_atoms: Number of atoms for Distributional RL
        :param v_min: Minimum value of the support for Distributional RL
        :param v_max: Maximum value of the support for Distributional RL
        """
        super(DuelingNetwork, self).__init__()

        self.is_noisy_nets = is_noisy_nets
        self.std_init = std_init
        self.is_distributional = is_distributional
        self.num_actions = num_actions
        self.num_atoms = num_atoms if is_distributional else 1
        self.v_min = v_min
        self.v_max = v_max

        # ✅ Shared feature extractor (Convolutional Encoder)
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        # ✅ Fully connected layer for shared feature representation
        self.fc1 = nn.Linear(32 * 4 * 4, 128)

        # ✅ Define Advantage and Value Streams
        if self.is_noisy_nets:
            self.advantage = NoisyLinear(128, num_actions * self.num_atoms, std_init=std_init)
            self.value = NoisyLinear(128, self.num_atoms, std_init=std_init)  # Value Stream only needs 1 output per atom
        else:
            self.advantage = nn.Linear(128, num_actions * self.num_atoms)
            self.value = nn.Linear(128, self.num_atoms)

        # ✅ Support for Distributional RL
        if self.is_distributional:
            self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Dueling Network."""
        # ✅ Convert from (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))

        # ✅ Compute Advantage & Value separately
        advantage = self.advantage(x)
        value = self.value(x)

        # ✅ Aggregate Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def reset_noise(self):
        """Reset Noisy Layers (for NoisyNets)"""
        if self.is_noisy_nets:
            self.advantage.reset_noise()
            self.value.reset_noise()


class DuelingDQNAgent:
    def __init__(self, env, lr=0.001, device="cpu", is_noisy_nets=False, std_init=0.4, is_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        """
        Initializes the Dueling DQN agent.
        """
        self.device = device

        # ✅ Use Dueling Q-Networks
        self.q = DuelingNetwork(env.observation_space.shape, env.action_space.n, is_noisy_nets, std_init, is_distributional, num_atoms, v_min, v_max).to(self.device)
        self.q_target = DuelingNetwork(env.observation_space.shape, env.action_space.n, is_noisy_nets, std_init, is_distributional, num_atoms, v_min, v_max).to(self.device)

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        # ✅ Copy initial weights to target network
        self.q_target.load_state_dict(self.q.state_dict())

    def update_target_network(self):
        """ ✅ Soft update target network """
        self.q_target.load_state_dict(self.q.state_dict())


def update_dueling_dqn(agent, gamma, obs, act, rew, next_obs, tm, is_double_dqn=False):
    """
    ✅ Update function for Dueling Q-Network.
    """
    agent.optimizer.zero_grad()

    with torch.no_grad():
        if is_double_dqn:
            # ✅ Double DQN: use the **online network** to get actions
            next_actions = agent.q(next_obs).argmax(dim=1, keepdim=True)
            target_q_values = agent.q_target(next_obs).gather(1, next_actions).squeeze(1)
        else:
            # ✅ Regular Dueling Q-Network
            target_q_values = agent.q_target(next_obs).max(dim=1)[0]

        # ✅ Compute TD target
        td_target = rew + gamma * target_q_values * (1 - tm.float())

    # ✅ Compute current Q-values
    q_values = agent.q(obs)
    q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

    # ✅ Compute loss & optimize
    loss = F.mse_loss(q_action, td_target)
    loss.backward()
    agent.optimizer.step()

    return loss.item()
