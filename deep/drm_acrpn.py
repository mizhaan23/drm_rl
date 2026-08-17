import time
import argparse
import torch
import numpy as np
import gymnasium as gym
import os
import inspect
import joblib
import random
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool

from algo import DRMACRPN
from policies import GaussianMLPPolicy, CategoricalMLPPolicy, CategoricalLinearPolicy
from utils import simulate_trajectories, discount_cumsum


def parse_args():
    parser = argparse.ArgumentParser()

    # environment specific args
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CliffWalking-v0",
                        help="the id of the gym environment")
    parser.add_argument("--env-seed", type=int, default=1,
                        help="the seed of the gym environment")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed of all rngs")

    # agent specific args
    parser.add_argument("--alpha", type=float, default=1e4,
                        help="the regularization parameter of the CRPN algorithm")
    parser.add_argument("--normalize-returns", type=lambda x: bool(strtobool(x)), default=True,
                        help="will normalize the returns by standard scaling")
    parser.add_argument("--hidden-sizes", type=eval, default=(),
                        help="hidden-sizes of the neural network")
    parser.add_argument("--drm-function", type=str, default="lambda x: x",  # torch.exp(-torch.sqrt(-torch.log(x)))
                        help="hidden-sizes of the neural network")

    # simulation specific args
    parser.add_argument("--max-timesteps", type=int, default=10000,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-updates", type=int, default=1000,
                        help="total update epochs for the policy")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="the number of parallel game environments")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be saved locally")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be tracked on wandb")

    # cuda stuff
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(args.alpha)}__{int(time.time())}"

    if args.track:
        import wandb

        run = wandb.init(
            # Set the project where this run will be logged
            project="rl-crpn-team-research",
            entity="rlwork",
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Check cuda or not
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(args.gym_id, is_slippery=False) for i in range(args.batch_size)],
        copy=False,
    )

    if not args.env_seed == -1:
        envs.reset(seed=args.env_seed)
        envs.action_space.seed(seed=args.env_seed)
        envs.observation_space.seed(seed=args.env_seed)

    discrete_state = False
    if isinstance(envs.single_observation_space, gym.spaces.Discrete):
        print(f"Discrete Observation Space for {args.gym_id}")
        discrete_state = True

    # Get Agent according to env action space type
    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        print(f"Discrete Action Space for {args.gym_id}")
        if len(tuple(args.hidden_sizes)) == 0:
            agent = CategoricalLinearPolicy(envs).to(device)
        else:
            agent = CategoricalMLPPolicy(envs, hidden_sizes=tuple(args.hidden_sizes), init_seed=args.seed).to(device)
    elif isinstance(envs.single_action_space, gym.spaces.Box):
        print(f"Continuous Action Space for {args.gym_id}")
        agent = GaussianMLPPolicy(envs, hidden_sizes=tuple(args.hidden_sizes), init_seed=args.seed).to(device)
    else:
        raise NotImplementedError("Unknown Action Space Type!")

    # save_path = f"{os.path.basename(__file__).rstrip('.py')}/{'__'.join(run_name.split('__')[:-1])}"
    print(args.drm_function)
    # agent.load_state_dict(torch.load(f"./data/drm_acrpn/Humanoid-v4__DRM_ACRPN1_dual__0__100000/trained_agent_params.pth", weights_only=True))

    optimizer = DRMACRPN(list(agent.parameters()), batch_size=args.batch_size, drm_function=eval(args.drm_function), alpha=args.alpha)  # 1 - (1-x) ** 1.

    # Simulation parameters
    max_timesteps = args.max_timesteps
    num_iterations = args.num_updates

    simulation_rewards = []
    simulation_rewards_ = []
    simulation_returns = []
    simulation_entropy = []

    start = time.time()
    for i in range(num_iterations):  # N

        t_ = time.time()
        traj_info, episodic_rewards, episodic_returns = simulate_trajectories(
            envs, policy=agent, critic=None, horizon=max_timesteps, device=device, discrete_state=discrete_state,
        )
        t1 = time.time() - t_

        # Agent learns here
        t_ = time.time()

        logP, R, dones = traj_info['log_probs'], traj_info['rewards'], traj_info['dones']
        Y = discount_cumsum(R, dones, gamma=1., normalize=False, device=device)
        # print(Y)

        Y0 = Y[0, :]
        Y0 = Y0 - Y0.min()  # similar to baseline and to convert Y0 to positive values
        l = (logP * ~dones).sum(0)  # log prob of trajectory
        # episodic_entropy = (-logP.exp() * logP * ~dones).sum(0)
        # print(episodic_entropy.mean())

        sorted_indices = torch.argsort(Y0)

        Y0_sorted = Y0[sorted_indices]
        l_sorted = l[sorted_indices]

        optimizer.zero_grad()
        optimizer.step(R_i=Y0_sorted, l_i=l_sorted)

        t2 = time.time() - t_

        simulation_rewards += episodic_rewards.tolist()
        simulation_returns += episodic_returns.tolist()
        # simulation_entropy += episodic_entropy.mean(axis=0).tolist()

        # avg_traj_length = dones.shape[0]

        print(f"Iteration {i}, Reward: {episodic_rewards.mean()}, Return: {episodic_returns.mean()}, T1: {np.round(t1, 3)}, "
              f"T2:{np.round(t2, 3)}", end="\n")

        if args.track:
            for t, r in enumerate(zip(episodic_rewards, episodic_returns)):
                wandb.log(
                    {
                        'rewards': r[0],
                        'returns': r[1],
                        'log_loss': l1[t].mean().item()
                    }
                )

    # Close environment
    envs.close()
    writer.close()

    end = time.time()
    print(f"TOTAL ELAPSED TIME: {end - start} seconds")

    if args.save:

        out_dict = {
            # simulation info
            "episodic_rewards": np.array(simulation_rewards, dtype=np.float32),
            # "episodic_rewards_": np.array(simulation_rewards_, dtype=np.float32),
            "episodic_returns": np.array(simulation_returns, dtype=np.float32),
            "episodic_entropy": np.array(simulation_entropy, dtype=np.float32),
            "batch_size": int(args.batch_size),
            "horizon": int(args.max_timesteps),
            "nits": int(args.num_updates),

            # agent info
            "agent_name": agent.__class__.__name__,
            "trained_agent_params": agent.state_dict(),
            "agent_input_keys": list(dict(inspect.signature(agent.__class__).parameters).keys()),

            # env info
            "env_id": str(args.gym_id),
        }
        save_path = f"{os.path.basename(__file__).rstrip('.py')}/{'__'.join(run_name.split('__')[:-1])}"
        if not os.path.exists(f'./data/{save_path}'):
            os.makedirs(f'./data/{save_path}')

        for k, v in out_dict.items():
            if k != "trained_agent_params":
                joblib.dump(v, f"./data/{save_path}/{k}.data")
            else:
                torch.save(v, f"./data/{save_path}/{k}.pth")
