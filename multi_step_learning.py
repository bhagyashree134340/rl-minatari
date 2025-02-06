import random
import torch
import torch.nn.functional as F
from collections import deque


class MultiStepBuffer:
    def __init__(self, maxlen, n_steps, gamma, device="cpu"):
        """
        Multi-step learning buffer for storing transitions.

        :param maxlen: Maximum buffer size
        :param n_steps: Number of steps for multi-step return
        :param gamma: Discount factor
        :param device: PyTorch device
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device
        self.buffer = deque(maxlen=maxlen)  # Use deque for efficient handling
        self.n_step_buffer = deque(maxlen=n_steps)  # Store temporary sequences

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

    def store(self, obs, action, reward, next_obs, terminated):
        """
        Stores a transition in the n-step buffer and computes the multi-step return.
        """
        #  Convert observations to PyTorch tensors before storing
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        
        self.n_step_buffer.append((obs, action, reward, next_obs, terminated))

        # If we have enough steps, compute the multi-step return
        if len(self.n_step_buffer) == self.n_steps:
            obs, action, discounted_reward, final_next_obs, final_terminated = self.get_multi_step_sample()
            self.buffer.append((obs, action, discounted_reward, final_next_obs, final_terminated))

    def get_multi_step_sample(self):
        """
        Computes multi-step return with n-step lookahead.
        """
        discounted_reward = 0.0
        for i, (obs, action, reward, next_obs, terminated) in enumerate(self.n_step_buffer):
            discounted_reward += (self.gamma ** i) * reward
            if terminated:
                break  # Stop adding rewards if episode ended
        
        #  Get the last next_obs and termination flag
        final_next_obs = self.n_step_buffer[-1][3]
        final_terminated = self.n_step_buffer[-1][4]

        # Remove the first transition
        first_obs, first_action, _, _, _ = self.n_step_buffer.popleft()

        return first_obs, first_action, discounted_reward, final_next_obs, final_terminated

    def sample(self, batch_size):
        """
        Sample a batch of transitions with replacement.
        Ensures that actions are converted to PyTorch tensors.
        """
        batch = random.choices(self.buffer, k=batch_size)
        
        #  Convert each element to PyTorch tensors
        obs, act, rew, next_obs, tm = zip(*batch)

        return (
            torch.stack(obs),  
            torch.tensor(act, dtype=torch.long, device=self.device),  #  Convert actions
            torch.tensor(rew, dtype=torch.float32, device=self.device),  #  Convert rewards
            torch.stack(next_obs),  
            torch.tensor(tm, dtype=torch.float32, device=self.device),  #  Convert termination flags
        )


def update_multi_step_dqn(q, q_target, optimizer, gamma, n_steps, obs, act, rew, next_obs, tm):
    """
    Update function for Multi-Step DQN.
    """
    optimizer.zero_grad()

    with torch.no_grad():
        #  Correct bootstrapped n-step target
        target_q_values = q_target(next_obs).max(dim=1)[0]

        td_target = rew + (gamma ** n_steps) * target_q_values * (1 - tm.float())

    q_values = q(obs)
    q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

    loss = F.mse_loss(q_action, td_target)
    loss.backward()
    optimizer.step()

    return loss.item()
