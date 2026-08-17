import numpy as np


class LinearRLAgent:
    def __init__(self, envs, discount_factor=0.99):
        self.envs = envs
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.batch_size = envs.num_envs
        self.discount_factor = discount_factor

    def get_state(self, *args, **kwargs):
        raise NotImplementedError('Method `get_state` not defined!')

    def get_action(self, *args, **kwargs):
        raise NotImplementedError('Method `get_action` not defined!')

    def get_value(self, *args, **kwargs):
        raise NotImplementedError('Method `get_value` not defined!')

    def learn(self, *args, **kwargs):
        raise NotImplementedError('Method `learn` not defined!')

    def td_update(self, *args, **kwargs):
        raise NotImplementedError('Method `td_update` not defined!')

    def compute_critic_h12(self, S, A, P, dQ, dones):
        nPhi = self.nPhi
        h12 = nPhi[..., np.newaxis] @ dQ[..., np.newaxis, :]
        h12 = h12 + np.transpose(h12, (0, 1, 3, 2))
        return h12.sum(axis=(0, 1)) / self.batch_size

    def compute_gradient_and_hessian_estimate(self, S, A, P, Y, dones, compute_hessian=False):

        A_ = self.one_hot_encode_actions(A)
        Phi = self._mult1(A_, S)  # returns phi(s, a)
        EPhi = self._mult1(P, S)  # returns E[phi(s, .)] according to `P`

        nPhi = Phi - EPhi  # normalized phi
        self.nPhi = nPhi
        grad_sum = self._mult1(1. * ~dones[..., np.newaxis], nPhi)
        grad_log_probs = self._mult1(Y[..., np.newaxis], nPhi)
        grad_est = np.sum(grad_log_probs, axis=(0, 1))
        grad_est = grad_est / self.batch_size

        hess_est = None
        if compute_hessian:
            # computing hessian estimate
            EPhiPhiT = self._mult2(self._diagonalize(P),
                                   S[..., np.newaxis] @ S[..., np.newaxis, :])  # returns E[Phi @ Phi.T]

            hess_est_1 = np.sum(grad_log_probs, axis=0)[..., np.newaxis] @ np.sum(grad_sum, axis=0)[..., np.newaxis, :]
            hess_est_1 = np.sum(hess_est_1, axis=0)

            hess_est_2 = self._mult2(Y[..., np.newaxis, np.newaxis],
                                     -EPhiPhiT + EPhi[..., np.newaxis] @ EPhi[..., np.newaxis, :])
            hess_est_2 = np.sum(hess_est_2, axis=(0, 1))
            hess_est = hess_est_1 + hess_est_2
            hess_est = hess_est / self.batch_size

        return grad_est, hess_est

    def one_hot_encode_actions(self, actions):
        a = actions.reshape(-1, order="F")
        b = np.zeros((a.size, self.action_space.n))
        b[np.arange(a.size), a] = 1
        B = b.reshape(actions.shape + (-1,), order="F")
        return B

    @staticmethod
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
                discounted_rewards[:, i] = (discounted_rewards[:, i] - discounted_rewards[:, i][:m].mean()) # \
                                           # / (discounted_rewards[:, i][:m].std() + 1e-9)
        return discounted_rewards * ~dones

    @staticmethod
    def compute_gae(rewards, values, next_values, dones, gamma, tau, normalize=True):
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

        if normalize:
            for i in range(rewards.shape[1]):
                m = np.argmax(dones[:, i]) - 1
                advantages[:, i] = (advantages[:, i] - advantages[:, i][:m].mean()) \
                                           / (advantages[:, i][:m].std() + 1e-9)

        return advantages * ~dones

    def normalize(self, matrix, dones):
        for i in range(matrix.shape[1]):
            m = np.argmax(dones[:, i]) - 1
            matrix[:, i] = (matrix[:, i] - matrix[:, i][:m].mean()) \
                               / (matrix[:, i][:m].std() + 1e-9)
        return matrix

    @staticmethod
    def _mult1(A, B):
        assert (A.shape[:-1] == B.shape[:-1]), f"{A.shape[:-1]}, {B.shape[:-1]}"
        out = np.einsum('...i,...j->...ij', A, B)
        out = out.reshape(A.shape[:-1] + (A.shape[-1] * B.shape[-1],), order="C")
        return out

    @staticmethod
    def _mult2(A, B):
        assert (A.shape[:-2] == A.shape[:-2]), f"{A.shape[:-2]}, {B.shape[:-2]}"
        out = np.einsum('...ij,...kl->...ikjl', A, B)
        out = out.reshape(A.shape[:-2] + (A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1]), order="C")
        return out

    @staticmethod
    def _diagonalize(A):
        A_ = np.stack(tuple([A] * A.shape[-1]), axis=-1)
        return A_ * np.eye(A.shape[-1])
