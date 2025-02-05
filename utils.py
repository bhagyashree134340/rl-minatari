import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from PIL import Image
import wandb

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
    def policy_fn(obs: torch.Tensor, epsilon: float = 0.0):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, num_actions)
        return Q(obs).argmax(dim=1).detach().numpy()[0]  # expects batch size 1
    return policy_fn


# def update_dqn(q, q_target, optimizer, gamma,
#                obs, act, rew, next_obs, tm, 
#                is_double_dqn, is_distributional, is_dueling_dqn, 
#                num_atoms, v_min, v_max):
#     """
#     Update the DQN, Double DQN, or Dueling DQN for one optimizer step using the target network.
#     """
#     optimizer.zero_grad()

#     if is_distributional:
#         with torch.no_grad():
#             _, next_pmfs = q_target.get_action(next_obs)
#             next_atoms = rew[:, None] + gamma * q_target.support[None, :] * (1 - tm[:, None].float())
            
#             # Projection step
#             delta_z = q_target.support[1] - q_target.support[0]
#             tz = next_atoms.clamp(v_min, v_max)

#             b = (tz - v_min) / delta_z
#             l = b.floor().clamp(0, num_atoms - 1)
#             u = b.ceil().clamp(0, num_atoms - 1)

#             d_m_l = (u + (l == u).float() - b) * next_pmfs
#             d_m_u = (b - l) * next_pmfs
#             target_pmfs = torch.zeros_like(next_pmfs)
            
#             for i in range(target_pmfs.size(0)):
#                 target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
#                 target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

#         _, old_pmfs = q.get_action(obs, act.flatten())
#         loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

#     else:
#         with torch.no_grad():
#             if is_double_dqn:
#                 next_actions = q(next_obs).argmax(dim=1, keepdim=True)
#                 target_values = q_target(next_obs).gather(1, next_actions).squeeze(1)
#             else:
#                 target_values = q_target(next_obs).max(dim=1)[0]

#             td_target = rew + gamma * target_values * (1 - tm.float())

#         # Gather Q-values for the taken actions
#         q_values = q(obs)
#         q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

#         # ✅ Apply dueling Q-learning if enabled
#         if is_dueling_dqn:
#             loss = F.smooth_l1_loss(q_action, td_target)
#         else:
#             loss = F.mse_loss(q_action, td_target)

#     optimizer.zero_grad()    
#     loss.backward()
#     optimizer.step()

#     return loss.item()  # for logging if desired

def update_dqn(q, q_target, optimizer, gamma, 
               obs, act, rew, next_obs, tm, 
               is_double_dqn, is_distributional, is_dueling_dqn, 
               num_atoms, v_min, v_max):
    """
    Update function for standard DQN, Double DQN, and optionally Dueling & Distributional DQN.
    """
    optimizer.zero_grad()

    if is_distributional:
        with torch.no_grad():
            _, next_pmfs = q_target.get_action(next_obs)
            next_atoms = rew[:, None] + gamma * q_target.support[None, :] * (1 - tm[:, None].float())
            delta_z = q_target.support[1] - q_target.support[0]
            tz = next_atoms.clamp(v_min, v_max)

            b = (tz - v_min) / delta_z
            l = b.floor().clamp(0, num_atoms - 1)
            u = b.ceil().clamp(0, num_atoms - 1)

            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = q.get_action(obs, act.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
    else:
        if is_double_dqn:
            with torch.no_grad():
                next_actions = q(next_obs).argmax(dim=1, keepdim=True)
                target_values = q_target(next_obs).gather(1, next_actions).squeeze(1)
                td_target = rew + gamma * target_values * (1 - tm.float())
        else:
            with torch.no_grad():
                td_target = rew + gamma * q_target(next_obs).max(dim=1)[0] * (1 - tm.float())

        q_values = q(obs)
        q_action = q_values.gather(1, act.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_action, td_target)

    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()

    return loss.item()




def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rgb_animation(rgb_arrays, filename, duration=50):
    """Save an animated GIF from a list of RGB arrays."""
    frames = []
    for rgb_array in rgb_arrays:
        rgb_array = (rgb_array * 255).astype(np.uint8)
        rgb_array = rgb_array.repeat(48, axis=0).repeat(48, axis=1)
        img = Image.fromarray(rgb_array)
        frames.append(img)
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def rendered_rollout(policy_fn, env, is_distributional, max_steps=10_000):
    """Rollout for one episode while saving all rendered images."""
    obs, _ = env.reset()
    imgs = [env.render()]

    for i in range(max_steps):
        if is_distributional:
            action, _ = policy_fn(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        else:
            action = policy_fn(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
        
        print(f"Step {i}: Action={action}, Valid Actions={list(range(env.action_space.n))}")
        obs, reward, terminated, truncated, _ = env.step(action)
        imgs.append(env.render())
        print(f"Step {i}: Action={action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")
        
        if terminated or truncated:
            break
    
    print(f"Number of frames collected: {len(imgs)}")
    return imgs


def animate(env, agent, is_noisy_nets, is_distributional, is_dueling_dqn, is_double_dqn, filename="trained.gif", max_steps=10_000):
    """Generates an animation of the agent interacting with the environment."""
    filename = (
        "trained_noisy_nets.gif" if is_noisy_nets 
        else "trained_distributional.gif" if is_distributional 
        else "trained_dueling_dqn.gif" if is_dueling_dqn
        else "trained_double_dqn.gif" if is_double_dqn 
        else "trained.gif"
    )

    def noisy_policy(obs: torch.Tensor):
        # return agent.q(obs).argmax(dim=1).detach().numpy()[0]
        if agent.is_dueling_dqn:
            return agent.dueling_agent.q(obs).argmax(dim=1).detach().numpy()[0]
        else:
            return agent.q(obs).argmax(dim=1).detach().numpy()[0]

    
    imgs = rendered_rollout(
        noisy_policy if is_noisy_nets 
        else agent.q.get_action if is_distributional 
        # else make_epsilon_greedy_policy(agent.q, num_actions=env.action_space.n),
        else make_epsilon_greedy_policy(
        agent.dueling_agent.q if agent.is_dueling_dqn else agent.q,  
        num_actions=env.action_space.n
        ),
        env, is_distributional, max_steps
    ) 
    # imgs = rendered_rollout(
    #     (lambda obs: agent.dueling_agent.q(obs).argmax(dim=1).detach().numpy()[0]) if is_noisy_nets and is_dueling_dqn
    #     else (lambda obs: agent.q(obs).argmax(dim=1).detach().numpy()[0]) if is_noisy_nets
    #     else agent.q.get_action if is_distributional
    #     else make_epsilon_greedy_policy(agent.q if not is_dueling_dqn else agent.dueling_agent.q, num_actions=env.action_space.n),
    #     env, is_distributional, max_steps
    # )
   

    run_dir = wandb.run.dir  
    gif_path = f"{run_dir}/{filename}"
    save_rgb_animation(imgs, gif_path)
    wandb.log({"animation": wandb.Video(gif_path, format="gif")})
    print(f"✅ Animation saved as {filename}")
