import time
import numpy as np
import torch
import torch.nn.functional as F

def _flatten(tensors):
    return np.concatenate([t.detach().cpu().numpy().ravel() for t in tensors])


# def _unflatten(flat: np.ndarray, template: List[Tensor]):
#     # print(template)
#     tensors = []
#     offset = 0
#     for t in template:
#         numel = t.numel()
#         shape = t.shape
#         chunk = flat[offset:offset + numel].reshape(shape)
#         tensors.append(torch.from_numpy(chunk).to(dtype=t.dtype, device=t.device))
#         offset += numel
#     return tensors
#
# def calculate_time(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"Time taken to execute {func.__name__}: {elapsed_time} seconds")
#         return result
#
#     return wrapper


def l1_distance_to_goal(state, shape=(4, 12), goal_state=47):
    # Convert flat indices to (row, col)
    row, col = np.unravel_index(state, shape)
    goal_row, goal_col = np.unravel_index(goal_state, shape)

    # Compute Manhattan (L1) distance
    return abs(row - goal_row) + abs(col - goal_col)

def simulate_trajectories(envs, policy, critic, horizon, device, discrete_state=False):
    n = envs.num_envs
    if discrete_state:
        observation_dim = (int(envs.single_observation_space.n),)
    else:
        observation_dim = envs.single_observation_space.shape

    # Initializing simulation matrices for the given batched episode
    observations = torch.zeros((horizon, n) + observation_dim, dtype=torch.float32).to(device)
    log_probs = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    entropies = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    rewards = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    rewards2 = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    # values = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    dones = torch.ones((horizon, n), dtype=bool).to(device)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    T = None

    count = 0
    for t in range(horizon):
        count += sum(obs == 47)
        obs = torch.tensor(np.float32(obs)).to(device)

        if discrete_state:
            obs = F.one_hot(obs.to(dtype=torch.long), num_classes=observation_dim[0]).to(dtype=torch.float32)

        action, log_prob, entropy = policy.get_action(obs)  # a ~ pi(s, .)
        if critic is not None:
            value = critic.get_value(obs)
            values[t] = value.to(device)

        # observations[t] = obs
        log_probs[t] = log_prob
        # entropies[t] = entropy
        dones[t] = torch.tensor(done).to(device)

        obs, reward, terminated, truncated, info = envs.step(action.cpu().detach().numpy())
        l1_dists = np.array([l1_distance_to_goal(obs_) for obs_ in obs])
        reward2 = reward - 0.5 * l1_dists  # eq 36

        done = done | (np.array(terminated) | np.array(truncated))
        # done = done | np.array(truncated)  # FOR FROZEN LAKE ONLY

        # Modify rewards to NOT consider data points after `done`
        # if sum(reward) > 1:
        #     print(np.float32(obs==15))
        #     print((np.float32(obs==15) == reward).mean())
            # print(sum(reward * ~terminated))
            # print(sum(reward * ~truncated))
            # print(reward)
            # print(truncated)
            # print(done)
        reward = reward * ~done
        reward2 = reward2 * ~done
        # r_ = np.where((reward == 0) & (np.array(terminated) == 1), -100, 0)
        # r_ += np.where((reward == 0) & (np.array(terminated) == 0), -1, 0)

        # FROZEN LAKE MODIFICATION
        # done = done | reward  # reward is 1 if goal state is reached
        # reward = r_ * ~done

        rewards[t] = torch.tensor(reward).to(device)
        rewards2[t] = torch.tensor(reward2).to(device)  # FOR CLIFFWALKING ONLY)

        if done.all():
            T = t+1
            break

    # For frozen lake modification only

    # print(count)
    cum_discounted_rewards = discount_cumsum(rewards, dones, gamma=0.99, normalize=False, device=device)
    mean_episode_return = torch.sum(cum_discounted_rewards, axis=0) / torch.sum(~dones, axis=0)

    traj_info = {
        # 'observations': observations[:T],
        'log_probs': log_probs[:T],
        # 'entropies': entropies[:T],
        'rewards': rewards2[:T],
        'rewards_': rewards[:T],
        # 'values': values[:T],
        'dones': dones[:T],
    }

    return traj_info, torch.sum(rewards, axis=0), torch.sum(rewards2, axis=0)


def simulate_transitions(envs, policy, critic, horizon, device):
    n = envs.num_envs

    # Initializing simulation matrices for the given batched episode
    log_probs = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    entropies = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    rewards = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    values = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    dones = torch.ones((horizon, n), dtype=bool).to(device)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    T = None

    for t in range(horizon):
        obs = torch.tensor(np.float32(obs)).to(device)

        action, log_prob, entropy = policy.get_action(obs)
        value = critic.get_value(obs)

        log_probs[t] = log_prob
        entropies[t] = entropy
        dones[t] = torch.tensor(done).to(device)

        obs, reward, terminated, truncated, info = envs.step(action.cpu().detach().numpy())
        done = (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`
        # reward = reward * ~truncated

        rewards[t] = torch.tensor(reward).to(device)
        values[t] = value.to(device)

        if done.all():
            T = t
            break

    cum_discounted_rewards = discount_cumsum(rewards, dones, gamma=0.99, normalize=False, device=device)
    mean_episode_return = torch.sum(cum_discounted_rewards, axis=0) / torch.sum(~dones, axis=0)

    traj_info = {
        'log_probs': log_probs[:T],
        'entropies': entropies[:T],
        'rewards': rewards[:T],
        'values': values[:T],
        'dones': dones[:T],
    }

    return traj_info, torch.sum(rewards, axis=0), mean_episode_return


@torch.jit.script
def discount_cumsum(rewards, dones, gamma: float, normalize: bool = True, device: torch.device = 'cpu') -> torch.Tensor:
    discounted_rewards = torch.zeros_like(rewards).to(device)
    cumulative_reward = torch.zeros_like(rewards[0]).to(device)
    t = -1
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * gamma  # Discount factor
        discounted_rewards[t, :] = cumulative_reward
        t -= 1
    if normalize:
        for i in range(rewards.shape[1]):
            m = torch.argmax(1. * dones[:, i]) - 1
            discounted_rewards[:, i] = (discounted_rewards[:, i] - discounted_rewards[:, i][:m].mean())  # / (
            #  discounted_rewards[:, i][:m].std() + 1e-9)
    return discounted_rewards * ~dones


def normalize(matrix, dones):
    for i in range(matrix.shape[1]):
        m = torch.argmax(1. * dones[:, i]) - 1
        matrix[:, i] = (matrix[:, i] - matrix[:, i][:m].mean())  #\
                       # / (matrix[:, i][:m].std() + 1e-9)
    return matrix


def compute_gae(rewards, values, next_values, dones, gamma, tau, device: torch.device = 'cpu'):
    """
    Computes the Generalized Advantage Estimate (GAE) for a batch of environments in parallel.

    Args:
    - rewards (Tensor): Tensor of rewards (horizon x n)
    - values (Tensor): Tensor of values (horizon x n)
    - next_values (Tensor): Tensor of next state values (horizon x n)
    - dones (Tensor): Tensor of done flags (horizon x n)
    - gamma (float): Discount factor
    - tau (float): GAE parameter

    Returns:
    - advantages (Tensor): Tensor of advantages (horizon x n)
    """
    # Ensure all inputs are of the same shape (horizon, n)
    horizon, n = rewards.shape

    # Initialize tensor to store advantages
    advantages = torch.zeros_like(rewards).to(device)

    # Compute deltas (TD errors) using the formula delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values * ~dones - values

    # Initialize gae (Generalized Advantage Estimate) for the last timestep
    gae = torch.zeros_like(rewards[0]).to(device)

    # Iterate backwards to compute GAE for each timestep
    for t in range(horizon - 1, -1, -1):
        # Compute the GAE recursively
        gae = deltas[t, :] + gamma * tau * ~dones[t, :] * gae
        advantages[t, :] = gae

    return advantages * ~dones
