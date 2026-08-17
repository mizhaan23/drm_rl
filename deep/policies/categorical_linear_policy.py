import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    np.random.seed(0)
    params = np.random.randn(layer.weight.data.numel(), ).astype(np.float32)
    layer.weight.data = 0 * torch.from_numpy(params.reshape(layer.weight.data.shape, order="C"))
    return layer


class CategoricalLinearPolicy(nn.Module):
    def __init__(
            self,
            envs,
            init_seed=None,
    ):
        super(CategoricalLinearPolicy, self).__init__()

        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            input_dim = envs.single_observation_space.n
        else:
            input_dim = np.prod(envs.single_observation_space.shape)

        output_dim = envs.single_action_space.n

        if init_seed is not None:
            torch.manual_seed(init_seed)

        self.logits = layer_init(nn.Linear(input_dim, output_dim, bias=False))

    def get_action(self, x):
        action_prob = F.softmax(self.logits(x), dim=-1)

        dist = Categorical(action_prob)
        # dist = Categorical(logits=self.logits(x))
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()  # action, log_prob


class CriticNetworkLinear(nn.Module):
    def __init__(
            self,
            envs,
            init_seed=None,
    ):
        super(CriticNetworkLinear, self).__init__()

        input_dim = np.prod(envs.single_observation_space.shape)
        output_dim = envs.single_action_space.n

        if init_seed is not None:
            torch.manual_seed(init_seed)

        self.critic_network = layer_init(nn.Linear(input_dim, 1, bias=False))

    def get_value(self, x):
        val = self.critic_network(x)
        return val.flatten()
