import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Tuple, Optional, Callable
import time
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize

def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to execute {func.__name__}: {elapsed_time} seconds")
        return result

    return wrapper


class DRMACRPN(Optimizer):
    def __init__(self,
                 params: List[Tensor],
                 alpha: float,
                 batch_size: int,
                 drm_function: Callable[float, float],
                 eps: float = 1e-8,
                 sigma: float = 1e-3,
                 maxiter: int = 100,
                 eta: float = 1e-4,
                 maximize: bool = False):

        if alpha < 0.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if maxiter < 1:
            raise ValueError(f"maxiter should be greater than or equal to 1 and not {maxiter}")

        defaults = dict(eps=eps, alpha=alpha, sigma=sigma, maxiter=maxiter, eta=eta,
                        maximize=maximize)

        f = drm_function

        # Number of points and the range
        x_values = torch.linspace(1, 0, batch_size + 1, requires_grad=True)[1:]

        # To store function values, first and second derivatives
        g = []
        g1 = []
        g2 = []

        for x in x_values:
            # Compute the function value at x
            f_val = f(x)
            g.append(f_val.item())

            # Compute the first derivative (gradient) of f at x
            grad_f = torch.autograd.grad(f_val, x, create_graph=True)[0]
            g1.append(grad_f.item())

            # Compute the second derivative (gradient of the gradient) of f at x
            if grad_f.requires_grad:
                double_grad_f = torch.autograd.grad(grad_f, x, allow_unused=True)[0]
                g2.append(double_grad_f.item())
            else:
                g2.append(0.)

        # Convert results to tensors or lists for display
        with torch.no_grad():
            self.g = torch.tensor(g, device='cuda:0')
            self.g1 = torch.tensor(g1, device='cuda:0')
            self.g2 = torch.tensor(g2, device='cuda:0')
            self.M_r = 0.

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)

    # @calculate_time
    def step(self, R_i, l_i, closure=None):  # Todo: Try incorporating the loses under closure
        '''
        # Prepare order statistics
        M_r = self.M_r

        c1 = torch.zeros_like(R_i)
        c2 = torch.zeros_like(R_i)

        c1[:-1] = (R_i[1:] - R_i[:-1]) * self.g1[:-1]
        c2[:-1] = (R_i[1:] - R_i[:-1]) * self.g2[:-1]

        c1[-1] = 0  # (R_i[-1] - M_r) * self.g1[-1]
        c2[-1] = 0  # (R_i[-1] - M_r) * self.g2[-1]

        psi = torch.cumsum(c1.flip(dims=[0]), dim=0).flip(dims=[0])

        l1 = psi * l_i
        l2 = l_i
        s1 = torch.cumsum(l_i, dim=0)
        s2 = -c2 * s1 / len(c2)
        '''

        # Prepare order statistics
        M_r = self.M_r

        c1 = torch.zeros_like(R_i)
        c2 = torch.zeros_like(R_i)

        c1[:-1] = (R_i[1:] - R_i[:-1]) * self.g1[:-1]
        c2[:-1] = (R_i[1:] - R_i[:-1]) * self.g2[:-1]

        c1[-1] = 0  # (R_i[-1] - M_r) * self.g1[-1]
        c2[-1] = 0  # (R_i[-1] - M_r) * self.g2[-1]

        psi2 = torch.cumsum(c2.flip(dims=[0]), dim=0).flip(dims=[0])
        psi = -R_i * self.g1
        psi[-1] = 0

        # print(c2)
        # print(psi2)
        # print(-(R_i - R_i.max())* self.g2)

        l1 = psi * l_i
        l2 = l_i
        # s1 = torch.cumsum(l_i, dim=0)
        # s2 = -c2 * s1 / len(c2)
        s1 = l_i
        s2 = -psi2 * l_i / len(c2)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        # Optimization logic here
        acrpn(
            params=list(group['params']),
            l1=l1,
            l2=l2,
            s1=s1,
            s2=s2,
            eps=group['eps'],
            alpha=group['alpha'],
            sigma=group['sigma'],
            maxiter=int(group['maxiter']),
            eta=group['eta'],
        )

        return loss


def acrpn(params: List[Tensor],
          l1: Tensor,
          l2: Tensor,
          s1: Tensor,
          s2: Tensor,
          eps: float,
          alpha: float,
          sigma: float,
          maxiter: int,
          eta: float):
    grad_estimates = list(torch.autograd.grad(l1.mean(), params, create_graph=True))  # \bar{g}
    grad_estimates_detached = list(g.detach() for g in grad_estimates)

    # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_estimate_hvp()`
    u1 = [torch.ones_like(l2, requires_grad=True)]
    ujp1 = list(torch.autograd.grad(l2, params, grad_outputs=u1, create_graph=True))

    u2 = [torch.ones_like(s2, requires_grad=True)]
    ujp2 = list(torch.autograd.grad(s2, params, grad_outputs=u2, create_graph=True))

    # Takes in a vector `v` and calculates the Hessian-vector product
    hvp_func = lambda v: torch._foreach_add(_estimate_hvp(params, v, l1, uJp=ujp1, u=u1, grad_estimates=grad_estimates),
                                            _estimate_hvp2(params, v, s1, uJp=ujp2, u=u2))

    delta, f_delta = cubic_subsolver(grad_estimates_detached, hvp_func, alpha, eps, sigma, maxiter, eta)

    if f_delta > -1 / 100 * math.sqrt(eps ** 3 / alpha):
        delta = cubic_finalsolver(grad_estimates_detached, hvp_func, alpha, eps, maxiter, eta, delta=delta)
    #
    # delta, f_delta = cubic_subsolver_(grad_estimates_detached, hvp_func, alpha, maxiter)

    # Update params
    with torch.no_grad():
        for i, param in enumerate(params):
            param.add_(delta[i], alpha=1.)
            # param.add_(grad_estimates_detached[i], alpha=-1e-3)


def cubic_subsolver_(grad_estimates_detached: List[Tensor],
                    hvp_func: Callable[[List[Tensor]], List[Tensor]],
                    alpha: float,
                    maxiter: int,
                    tol=1e-6):

    g = _flatten(grad_estimates_detached)

    def HVP(v):
        return _flatten(hvp_func(_unflatten(v, grad_estimates_detached)))

    def _fg(p, hvp, g, alpha):
        n = np.linalg.norm(p)
        Hp = hvp(p)
        s = np.dot(g, p) + .5 * np.dot(Hp, p) + alpha / 6 * n ** 3
        j = g + Hp + alpha / 2 * n * s
        # print(np.linalg.norm(g))
        print(np.linalg.norm(j))
        return s, j

    def _hessx(p, x, hvp, g, alpha):
        n = np.linalg.norm(p)
        Hx = hvp(x)
        return Hx + alpha / 2 * (np.dot(p, x) * p / n + n * x)

    # print(g)
    # v0 = -np.sqrt(2 / alpha / np.linalg.norm(g)) * g * 0.
    v0 = - 0.01 * g / np.linalg.norm(g)
    # return _unflatten(v0, grad_estimates_detached), None
    result = minimize(
        _fg, v0, method='Newton-CG', jac=True, hessp=_hessx, args=(HVP, g, alpha),
        tol=np.finfo(np.float32).eps, options={'maxiter': 500}
    )
    print(np.dot(result.x, v0) / np.linalg.norm(result.x) / np.linalg.norm(v0))
    print(np.linalg.norm(result.x - v0))
    delta = _unflatten(result.x, grad_estimates_detached)
    return delta, None

def _flatten(tensors):
    return np.concatenate([t.detach().cpu().numpy().ravel() for t in tensors])


def _unflatten(flat: np.ndarray, template: List[Tensor]):
    # print(template)
    tensors = []
    offset = 0
    for t in template:
        numel = t.numel()
        shape = t.shape
        chunk = flat[offset:offset + numel].reshape(shape)
        tensors.append(torch.from_numpy(chunk).to(dtype=t.dtype, device=t.device))
        offset += numel
    return tensors



def cubic_subsolver(grad_estimates_detached: List[Tensor],
                    hvp_func: Callable[[List[Tensor]], List[Tensor]],
                    alpha: float,
                    eps: float,
                    sigma: float,
                    maxiter: int,
                    eta: float):
    """
    Implementation of the cubic-subsolver regime as described in Carmon & Duchi paper

    define: f(x; g) = 1/2 <x, H[x]> + <g, x> + alpha / 6 ||x||^3 [Cubic sub-model]
            f'(x; g) = H[x] + g + alpha / 2 ||x|| x
            threshold = -1 / 100 * eps^(3/2) * alpha^(-1/2)

    ** Cauchy step **
    R_c  <-  -beta + sqrt( beta^2 + 2*||g|| / alpha), where beta = <g, Hg> / (alpha * ||g||^2)
    Delta  <-  -R_c * g / ||g||

    if f(Delta) < threshold:
        return Delta, f(Delta)
    else:
        GO TO GRADIENT DESCENT

    ** Gradient Descent **

    g_noise <- g + sigma * q, where q ~ Uniformly from a sphere

    ** Cauchy step with noise **
    R_c_noise  <-  -beta_noise + sqrt( beta_noise^2 + 2*||g_noise|| / alpha),
    where beta_noise = <g_noise, H g_noise> / (alpha * ||g_noise||^2)

    Delta_0  <-  -R_c_noise * g_noise / ||g_noise||

    for 1, 2, ..., maxiter:
        Delta_t <- Delta_{t-1} - eta * (g_noise + H[g_noise] + alpha / 2 * ||Delta_{t-1}|| * Delta_{t-1})

        if f(Delta_t; g) < threshold:
            return Delta_t, f(Delta_t; g)

    return Delta_t, f(Delta_t)
    """

    # Take Cauchy-Step
    grad_norm = _compute_norm(grad_estimates_detached)

    beta = _compute_dot_product(grad_estimates_detached, hvp_func(grad_estimates_detached))
    beta /= grad_norm * grad_norm
    # print(beta)
    # beta /= alpha
    beta = 0.

    R_c = -beta + math.sqrt(beta * beta + 2 * grad_norm / alpha)

    # Cauchy point
    delta = list(-R_c * g_detached / grad_norm for g_detached in grad_estimates_detached)

    # sub-model value at delta_0 (Cauchy-point), i.e. f(delta_0)
    # where f(x) = 1/2 <x, H[x]> + <g, x> + alpha / 6 ||x||^3 [Cubic sub-model]
    # -> f(delta_0) = - 1/2 * (R_c ||g|| + alpha / 6 * R_c^3)
    f_delta = - 1 / 2 * (R_c * grad_norm + alpha / 6 * R_c ** 3)

    if f_delta < -1 / 100 * math.sqrt(eps ** 3 / alpha):
        return delta, f_delta
    print('sub-solver gd')
    # If above condition is not satisfied, try noisy gradient descent
    q = list(torch.randn(g_detach.shape, device=g_detach.device) for g_detach in grad_estimates_detached)
    q_norm = _compute_norm(q) + 1e-9

    grad_noise = list(g_detach + sigma * q_ / q_norm for g_detach, q_ in zip(grad_estimates_detached, q))

    # Take Cauchy-Step with noisy gradient
    grad_noise_norm = _compute_norm(grad_noise)

    beta_noise = _compute_dot_product(grad_noise, hvp_func(grad_noise))
    beta_noise /= alpha * grad_noise_norm * grad_noise_norm
    R_c_noise = -beta_noise + math.sqrt(beta_noise * beta_noise + 2 * grad_noise_norm / alpha)

    # Cauchy point w noisy grad
    delta = list(-R_c_noise * g_noise / grad_noise_norm for g_noise in grad_noise)

    hvp_delta = hvp_func(delta)
    norm_delta = _compute_norm(delta)

    for _ in range(maxiter):
        # Update delta in-place
        for i, grad_noise_i in enumerate(grad_noise):
            delta[i][:] -= eta * (grad_noise_i + hvp_delta[i] + alpha / 2 * norm_delta * delta[i])

        hvp_delta = hvp_func(delta)
        norm_delta = _compute_norm(delta)

        # sub-model value at delta
        f_delta = (0.5 * _compute_dot_product(delta, hvp_delta) +
                   _compute_dot_product(grad_estimates_detached, delta) +
                   alpha / 6 * norm_delta ** 3)

        # If condition is reached, break early
        if f_delta < -1 / 100 * math.sqrt(eps ** 3 / alpha):
            return delta, f_delta

    return delta, f_delta


def cubic_finalsolver(grad_estimates_detached: List[Tensor],
                      hvp_func: Callable[[List[Tensor]], List[Tensor]],
                      alpha: float,
                      eps: float,
                      maxiter: int,
                      eta: float,
                      delta: List[Tensor]):
    """
    TODO : Add description
    """
    print('final-solver gd')
    # Start from cauchy point, i.e. delta = delta
    grad_iterate = deepcopy(grad_estimates_detached)
    for _ in range(maxiter):
        for i, g_i in enumerate(grad_iterate):
            delta[i][:] -= eta * g_i

        hvp_delta = hvp_func(delta)
        norm_delta = _compute_norm(delta)
        for i, g_detach_i in enumerate(grad_estimates_detached):
            grad_iterate[i][:] = g_detach_i + hvp_delta[i] + alpha / 2 * norm_delta * delta[i]

        norm_grad_iterate = _compute_norm(grad_iterate)
        if norm_grad_iterate < eps / 2:
            break

    return delta


def _estimate_hvp(params: List[Tensor],
                  v: List[Tensor],
                  l1: Tensor,
                  uJp: List[Tensor],
                  u: List[Tensor],
                  grad_estimates: List[Optional[Tensor]]):
    """
    Main function that calculates the Hessian-vector product efficiently, using the trick mentioned in the main paper.
    """
    Jvp = torch.autograd.grad(uJp, u, grad_outputs=v, retain_graph=True)[0]  # using fwd autodiff trick
    hvp1 = list(torch.autograd.grad(l1, params, grad_outputs=Jvp / len(Jvp), retain_graph=True))

    # TODO: Try using ``_compute_dot_product`` here instead
    gTv = sum((g_ * v_).sum() for g_, v_ in zip(grad_estimates, v))  # inner product <grad, v>
    hvp2 = list(torch.autograd.grad(gTv, params, retain_graph=True))

    return list(h1 + h2 for h1, h2 in zip(hvp1, hvp2))


def _estimate_hvp2(params: List[Tensor],
                   v: List[Tensor],
                   l1: Tensor,
                   uJp: List[Tensor],
                   u: List[Tensor]):
    """
    Main function that calculates the Hessian-vector product efficiently, using the trick mentioned in the main paper.
    """
    Jvp = torch.autograd.grad(uJp, u, grad_outputs=v, retain_graph=True)[0]  # using fwd autodiff trick
    return list(torch.autograd.grad(l1, params, grad_outputs=Jvp / len(Jvp), retain_graph=True))


def _compute_dot_product(a: List[Optional[torch.Tensor]], b: List[Optional[torch.Tensor]]) -> float:
    return torch.tensor([(a_.detach() * b_.detach()).sum() for a_, b_ in zip(a, b)]).sum()


def _compute_norm(a: List[Optional[torch.Tensor]]) -> float:
    return math.sqrt((torch.tensor([(a_ * a_).sum() for a_ in a])).sum())
