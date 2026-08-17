import time
import numpy as np
import torch
import gymnasium as gym


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to execute {func.__name__}: {elapsed_time} seconds")
        return result

    return wrapper


def l1_distance_to_goal(state, shape=(4, 12), goal_state=47):
    # Convert flat indices to (row, col)
    row, col = np.unravel_index(state, shape)
    goal_row, goal_col = np.unravel_index(goal_state, shape)

    # Compute Manhattan (L1) distance
    return abs(row - goal_row) + abs(col - goal_col)


def simulate_trajectories(envs, agent, policy, horizon):
    n = envs.num_envs

    observation_dim = envs.single_observation_space.shape
    state_dim = (agent.num_features,)
    action_dim = envs.single_action_space.shape
    num_actions = (envs.single_action_space.n,)  # number of discrete actions

    # Initializing simulation matrices for the given batched episode
    observations = np.zeros((horizon, n) + observation_dim, dtype=np.float32)
    states = np.zeros((horizon, n) + state_dim, dtype=np.float32)
    actions = np.zeros((horizon, n) + action_dim, dtype=np.int32)
    action_probs = np.zeros((horizon, n) + num_actions, dtype=np.float32)
    rewards = np.zeros((horizon, n), dtype=np.float32)
    rewards_ = np.zeros((horizon, n), dtype=np.float32)
    dones = np.ones((horizon, n), dtype=bool)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    m = None
    for t in range(horizon):

        state = agent.get_state(obs)  # (bs, state_dim)
        action, action_prob = policy(obs)  # (bs, ), (bs, action_dim)

        observations[t] = obs
        states[t] = state
        actions[t] = action
        action_probs[t] = action_prob
        dones[t] = done

        obs, reward, terminated, truncated, info = envs.step(action)
        l1_dists = np.array([l1_distance_to_goal(obs_) for obs_ in obs])
        reward_ = reward - 0.5 * l1_dists
        done = done | (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`

        reward = reward * ~done
        reward_ = reward_ * ~done
        rewards[t] = reward
        rewards_[t] = reward_

        if done.all():
            m = t
            break

    cum_discounted_rewards = agent.discount_cumsum(rewards, dones, gamma=0.99, normalize=False)
    cum_discounted_rewards = np.array(cum_discounted_rewards).astype(np.float32)
    mean_episode_return = np.sum(cum_discounted_rewards, axis=0) / np.sum(~dones, axis=0)

    traj_info = {
        'observations': observations[:m],
        'states': states[:m],
        'actions': actions[:m],
        'action_probs': action_probs[:m],
        'rewards': rewards[:m],
        'rewards_': rewards_[:m],
        'cum_discounted_rewards': cum_discounted_rewards[:m],
        'mean_episode_return': mean_episode_return,
    }
    return traj_info, dones[:m], np.sum(rewards, axis=0), mean_episode_return


def simulate_transitions(envs, agent, policy, horizon):
    n = envs.num_envs
    observation_dim = envs.single_observation_space.shape
    state_dim = (agent.num_features,)
    action_dim = envs.single_action_space.shape
    num_actions = (envs.single_action_space.n,)  # number of discrete actions

    # Initializing simulation matrices for the given batched episode
    observations = np.zeros((horizon, n) + observation_dim, dtype=np.float32)
    states = np.zeros((horizon, n) + state_dim, dtype=np.float32)
    actions = np.zeros((horizon, n) + action_dim, dtype=np.int32)
    action_probs = np.zeros((horizon, n) + num_actions, dtype=np.float32)
    rewards = np.zeros((horizon, n), dtype=np.float32)
    # values = np.zeros((horizon, n), dtype=np.float32)
    dones = np.ones((horizon, n), dtype=bool)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    m = None
    for t in range(horizon):
        state = agent.get_state(obs)  # (bs, state_dim)
        action, action_prob = policy(obs)  # (bs, ), (bs, action_dim)
        # value = agent.get_value(state)

        observations[t] = obs
        states[t] = state
        actions[t] = action
        action_probs[t] = action_prob
        dones[t] = done

        obs, reward, terminated, truncated, info = envs.step(action)
        done = (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`
        reward = reward * ~done
        rewards[t] = reward
        # values[t] = value

        if done.all():
            m = t
            # break

    cum_discounted_rewards = agent.discount_cumsum(rewards, dones, gamma=0.99, normalize=False)
    cum_discounted_rewards = np.array(cum_discounted_rewards).astype(np.float32)
    mean_episode_return = np.sum(cum_discounted_rewards, axis=0) / np.sum(~dones, axis=0)

    traj_info = {
        'observations': observations,
        'states': states,
        'actions': actions,
        'action_probs': action_probs,
        'rewards': rewards,
        'cum_discounted_rewards': cum_discounted_rewards,
        'mean_episode_return': mean_episode_return,
    }

    return traj_info, dones, np.sum(rewards, axis=0), mean_episode_return


def discount_cumsum(rewards, dones, gamma, normalize=True):
    discounted_rewards = np.zeros_like(rewards)
    cumulative_reward = np.zeros_like(rewards[0])
    t = -1
    for r in rewards[::-1]:
        cumulative_reward = r + cumulative_reward * gamma  # Discount factor
        discounted_rewards[t, :] = cumulative_reward.copy()
        t -= 1
    if normalize:
        for i in range(rewards.shape[1]):
            m = np.argmax(dones[:, i]) - 1
            discounted_rewards[:, i] = (discounted_rewards[:, i] - discounted_rewards[:, i][:m].mean()) \
                                       / (discounted_rewards[:, i][:m].std() + 1e-9)
    return discounted_rewards * ~dones


def compute_gae(rewards, values, next_values, dones, gamma, tau):
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
    advantages = np.zeros_like(rewards)

    # Compute deltas (TD errors) using the formula delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values * ~dones - values

    # Initialize gae (Generalized Advantage Estimate) for the last timestep
    gae = np.zeros_like(rewards[0])

    # Iterate backwards to compute GAE for each timestep
    for t in range(horizon - 1, -1, -1):
        # Compute the GAE recursively
        gae = deltas[t, :] + gamma * tau * ~dones[t, :] * gae
        advantages[t, :] = gae

    return advantages * ~dones
