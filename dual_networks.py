
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, obs_shape, num_actions):
        """
        Implements Dueling Q-Network.
        :param obs_shape: Observation space shape
        :param num_actions: Number of actions
        """
        super(DuelingDQN, self).__init__()

        # ✅ Feature extractor (shared encoder)
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        
        # ✅ Fully connected layers for Advantage & Value Streams
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        
        # ✅ Separate Streams: Advantage & Value
        self.advantage = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)  # Only one output for V(s)

    def forward(self, x):
        # ✅ Change shape from (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        # ✅ Compute Advantage & Value separately
        advantage = self.advantage(x)
        value = self.value(x)

        # ✅ Aggregate Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class DuelingDQNAgent:
    def __init__(self, env, lr=0.001, device="cpu"):
        """
        Initializes the Dueling DQN agent.
        """
        self.device = device

        # ✅ Use Dueling Q-Networks
        self.q = DuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.q_target = DuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        # ✅ Copy initial weights to target network
        self.q_target.load_state_dict(self.q.state_dict())

    def update_target_network(self):
        """ ✅ Soft update target network """
        self.q_target.load_state_dict(self.q.state_dict())


def update_dueling_dqn(agent, gamma, obs, act, rew, next_obs, tm):
    """
    ✅ Update function for Dueling DQN.
    """
    agent.optimizer.zero_grad()

    with torch.no_grad():
        # ✅ Compute next state Q-values
        next_q_values = agent.q_target(next_obs)
        next_actions = next_q_values.argmax(dim=1, keepdim=True)

        # ✅ Get max Q-value from the target network
        target_q_values = agent.q_target(next_obs).gather(1, next_actions).squeeze(1)

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
