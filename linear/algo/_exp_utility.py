import numpy as np
import gymnasium as gym
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import softmax
from scipy.optimize import minimize
from ._agent import LinearRLAgent

ALPHA = 0.01
BETA = 0.99
THETA = 1


class ExpUtility(LinearRLAgent):
    def __init__(self, envs, alpha=1e4, normalize_returns=False, poly_degree=1, set_bias=False):
        '''
        Implementation of Eqn (9) from the draft.
        :param envs:
        :param normalize_returns:
        :param poly_degree:
        :param set_bias:
        '''
        super().__init__(envs)

        self.alpha = alpha  # learning rate
        self.beta = BETA  # beta factor
        self.theta = THETA  # temperature

        if isinstance(envs.single_observation_space, gym.spaces.Discrete):
            self.discrete_state = True
            self.num_features = self.observation_space.n
        else:
            self.featurize = PolynomialFeatures(degree=poly_degree, include_bias=set_bias, order='F')
            self.featurize.fit(self.observation_space.sample().reshape(1, -1))

            self.num_features = self.featurize.n_output_features_

        self.num_params = self.num_features * self.action_space.n

        # np.random.seed(1234)

        self.params = np.random.randn(self.num_params, ).astype(np.float32) * 0

        self.x_table = np.ones((self.num_features, self.action_space.n)).astype(np.float32)
        self.q_table = np.zeros((self.num_features, self.action_space.n)).astype(np.float32)

        # self.value_params = np.random.randn(self.num_features, ).astype(np.float32)  # value params
        self.normalize_returns = normalize_returns

    def get_state(self, observation):
        if self.discrete_state:
            one_hot = np.zeros((observation.size, self.num_features), dtype=int)
            one_hot[np.arange(observation.size), observation] = 1
            return one_hot
        return self.featurize.transform(observation)  # (bs, state_dim)

    def get_value(self, params, state):
        p = params.reshape((self.num_features,), order="F")  # (state_dim,)
        return state @ p  # (bs, state_dim) x (state_dim,)

    def get_values(self, params, S, R, dones):
        V = np.zeros_like(R)
        for t in range(S.shape[0]):
            s = S[t, :, :]  # (bs, statedim)
            v = self.get_value(params, s)  # (bs,)
            V[t] = v
        return V * ~dones



    def get_x_values(self, O, A=None, dones=None):
        X = np.zeros_like(O)
        for t in range(O.shape[0]):
            ot = O[t, :]
            if A is not None:
                at = A[t, :]
                x = []
                for o, a in zip(ot, at):
                    x.append(self.x_table[int(o), int(a)])
                X[t] = x
            else:
                x = []
                for o in ot:
                    x.append(self.x_table[int(o), :].min())
                X[t] = x

        return X * ~dones


    def get_action(self, state):
        p = self.params.reshape((self.num_features, self.action_space.n), order="F")  # (state_dim, num_actions)
        logits = state @ p  # (bs, state_dim) x (state_dim, num_actions)
        action_prob = softmax(logits, axis=1)  # (bs, num_actions)

        cdf = np.cumsum(action_prob, axis=1)  # (bs, num_actions)
        rvs = np.random.rand(action_prob.shape[0], )  # (bs, )

        # Sample action
        action = np.argmax(rvs[:, np.newaxis] < cdf, axis=1)
        return action, action_prob

    def td_update(self, params, S, R, dones, gamma, tau, beta):
        # Critic / Value update
        x = np.ones_like(params) * 1000
        its = 0
        while (np.abs(x).mean() > 0.0001) and (its <= 1000):
            # Get value at S
            V = self.get_values(params, S, R, dones)
            nextV = np.zeros_like(V)
            nextV[:-1, :] = V[1:, :]
            nextV = nextV * ~dones

            # TD(lambda) update
            delta = self.compute_gae(R, V, nextV, dones, gamma=gamma, tau=tau, normalize=False)
            # print(delta.shape, S.shape)
            x = (delta[..., np.newaxis] * S).sum((0, 1)) / np.prod(delta.shape[:2])
            params[:] = params + beta * x
            its += 1
        return delta, V, nextV


    def _eqn9(self, S, A, R):
        return

    def learn(self, traj_info, dones, flatten=False, traj_info_perturb=None, dones_perturb=None, nu=None, u=None,
              alp=None):
        # GAMMA = self.discount_factor

        S = traj_info['states']
        A = traj_info['actions']
        P = traj_info['action_probs']
        R = traj_info['rewards']
        # Y = self.discount_cumsum(R, dones, gamma=GAMMA, normalize=False)

        O = traj_info['observations']

        X = self.get_x_values(O, A, dones)[:-1, :]
        nextX = self.get_x_values(O[1:, :], None, dones[1:,])

        # Eqn 9 update
        e = - self.theta / self.beta * R[:-1, :] + self.beta * nextX.log()
        self.alpha * (e.exp() - X)
        print(X.shape)
        exit()


        # Critic / Value update
        delta, V, nextV = self.td_update(self.value_params, S, R, dones, gamma=0.99, tau=0, beta=0.001)
        q = delta + V
        # q = q - q[0].min()
        # q = delta

        if flatten:
            S = S.reshape(1, -1, S.shape[2])
            A = A.reshape(1, -1)
            P = P.reshape(1, -1, P.shape[2])
            R = R.reshape(1, -1)
            Y = Y.reshape(1, -1)
            q = q.reshape(1, -1)
            dones = dones.reshape(1, -1)

            print(np.log(np.linalg.norm(h12)))
        # Y = self.normalize(Y, dones)

        # Y = self.discount_cumsum(R, dones, gamma=GAMMA, normalize=self.normalize_returns)
        # Y = self.compute_gae(R, V, nextV, dones, gamma=GAMMA, tau=LAMBDA, normalize=False) + V

        # Y_ = self.discount_cumsum(R, dones, gamma=GAMMA, normalize=False)
        # print(np.mean(np.abs(Y-Y_)))

        # Compute the optima for the cubic-regularized sub-problem
        # v0 = np.random.randn(self.num_params, )
        v0 = self.cauchy_point(H, g, ALPHA)
        self.params[:] = self.params + v0
        return g, H, None

        result = minimize(
            self._fg, v0, method='Newton-CG', jac=True, hess=self._hess, args=(H, g, ALPHA),
            options={'maxiter': 1000}
        )
        #
        # print(result.success)
        # print(result.message)
        # print(np.abs(v0 - result.x).mean())
        # make update
        self.params[:] = self.params + result.x
        return g, H, None

    @staticmethod
    def _fg(v, H, g, alpha):
        n = np.linalg.norm(v)
        Hv = H @ v
        s = np.dot(g, v) + .5 * np.dot(Hv, v) + alpha / 6 * n ** 3
        j = g + Hv + alpha / 2 * n * v
        return s, j

    @staticmethod
    def _hess(v, H, g, alpha):
        n = np.linalg.norm(v)
        return H + alpha / 2 * (v @ v.T / n + n * np.eye(len(v)))

    @staticmethod
    def cauchy_point(H, g, alpha):
        grad_norm = np.linalg.norm(g)
        if H is None:
            return - np.sqrt(2 / alpha / grad_norm) * g
        beta = np.dot(g, H @ g)
        beta /= grad_norm**2
        beta /= alpha
        r_c = -beta + np.sqrt(beta * beta + 2 * grad_norm / alpha)
        return -r_c / grad_norm * g

