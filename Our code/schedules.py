from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

# start with computation helpers for EB

"""
EB (equation 5) from the paper:
    EB = C_S * sum_k h_k * eps(t_k)  +  C_L * sum_k h_k^2 * L(t_k)

where: 
h_k = t_k - t_{k-1} = step size
eps(t) = flow matching error at t = (FME, analogue of score matching loss)
L(t) = smoothness term

Note:
for RF with straight paths, scoere smoothness is roughly constant. 
the velocity field is trained to be linear, we default to L(t) = 1.
Lipschtiz setting - can also pass L(t) = 1/sigm_t^4 (early stopping)

E(t, s) shorthand (from section 4 of paper):
    E(t, s) = C_S * eps(t) * (s - t)  +  C_L * L(t) * (s - t)^2

""" 

def compute_E(
    t: float,
    s: float, 
    eps_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    C: float = 1.0,
) -> float:

    """
    compute E(t, s) = eps(t)*(s-t) + C * L(t)*(s-t)^2
    C is C_L / C_S : ratio hyperparameter. We are working in unts where C_S = 1.
    """

    h = s - t
    return eps_fn(t) * h + C * L_fn(t) * h * h

def compute_EB(
    schedule: List[float],
    eps_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    C: float = 1.0,
) -> float: 
    
    """
    COmpute the total EB for a given schedule. 
    Schedule: [t_0 = 0, t_1, .., t_N = 1]
    """

    total = 0.0
    for k in range(1, len(schedule)):
        total += compute_E(schedule[k - 1], schedule[k], eps_fn, L_fn, C)
    return total

# default smoothness functions

def L_lipschitz(_t: float) -> float:
    """ LIpschitze setting: L(t) = 1 (constant)."""
    return 1.0

def L_early_stopping(t: float, eps: float = 1e-3) -> float:
    
    """
    early stopping setting: L(t) = 1 / sigma_t^4.
    for RF: sigma_t ~ sqrt(t*(1-t)), so we use sigma_t = sqrt(t).
    We clamp t away from 0, avoiding division by zero.
    """

    t_safe = max(t, eps)
    # variance will grow linearly with t in RF
    sigma_sq = t_safe
    return 1.0 / (sigma_sq ** 2 + eps)

# GA implementation (Algorithm 1)

class GradientAdjustingSchedule:

    """
    this is a continuous-time adaptive schedule via gradient descent on EB.
    For RF, eps(t) can be evaluated at any t in [0,1] by running a small
    Monte Carlo estimate with the trained network.

    Parameters:
    init_schedule : list of float
        Initial time points [t_0=0, ..., t_N=1]. Interior points are optimized.
    
    eps_fn: callable float -> float
        Flow matching error at time t (estimated from the network).
    
    L_fn: callable float -> float
        Smoothness function. Default: Lipschitz (constant = 1).
    
    C: float
        Ratio C_L / C_S (hyperparameter, default 1.0).
    
    lr: float
        Learning rate for gradient descent on interior points.
    
    n_iters: int
        Number of gradient descent iterations.
    
    min_gap: float
        Minimum gap enforced between adjacent points (prevents collapse).
    """

    def __init__(
        self, 
        init_schedule: List[float],
        eps_fn: Callable[[float], float],
        L_fn: Callable[[float], float] = L_lipschitz,
        C: float = 1.0,
        lr: float = 1e-3,
        n_iters: int = 200,
        min_gap: float = 1e-3,
    ):
        self.eps_fn = eps_fn
        self.L_fn = L_fn
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.min_gap = min_gap

        # interior points only (t_0 and t_N are fixed)
        self.t0 = init_schedule[0]
        self.tN = init_schedule[-1]

        # mutable interior points
        self._points = list(init_schedule[1: -1])
    
    @property
    def schedule(self) -> List[float]:
        return [self.t0] + sorted(self._points) + [self.tN]
    
    def _numerical_gradient(self, idx: int, delta: float = 1e-4) -> float:
        # finite difference gradient of EB wrt interior point idx
        pts = self._points[:]

        # eb depends only on E(t_{k-1}, t_k) + E(t_k, t_{k+1})
        # this is the full schedule for two neighbors
        full = self.schedule
        #index in full schedule
        k = idx + 1
        t_prev = full[k - 1]
        t_next = full[k + 1]
        t_k = full[k]

        def local_eb(t) -> float:
            return (
                compute_E(t_prev, t, self.eps_fn, self.L_fn, self.C)
              + compute_E(t, t_next, self.eps_fn, self.L_fn, self.C)
            )
        return (local_eb(t_k + delta) - local_eb(t_k - delta)) / (2 * delta)
    
    def run(self, verbose: bool = True) -> Tuple[List[float], List[float]]:
        """
        gonna run gradient descent.
        
        Returns: 
        
        final_schedule: sorted list of time points.
        eb_history: eb value at every iteration
        """

        eb_history = [compute_EB(self.schedule, self.eps_fn, self.L_fn, self.C)]
        for iteration in range(self.n_iters):
            grads = [self._numerical_gradient(i) for i in range(len(self._points))]

            # gradient step
            new_pts = [
                self._points[i] - self.lr * grads[i]
                for i in range(len(self._points))

            ]

            # sort and project: enfore t_0 < t_1 < ... < t_N with min_gap
            new_pts = sorted(new_pts)
            new_pts = self._project(new_pts)
            self._points = new_pts

            eb = compute_EB(self.schedule, self.eps_fn, self.L_fn, self.C)
            eb_history.append(eb)

            if verbose and (iteration + 1) % 20 == 0:
                print(f" GA iter {iteration+1:4d} | EB = {eb:.6f}")

        return self.schedule, eb_history
    
    def _project(self, pts: List[float]) -> List[float]:
        # enforce ordering with minimum gap from t0, tN, and each other

        result = []
        lo = self.t0 + self.min_gap
        for p in pts:
            p = max(p, lo)
            p = min(p, self.tN - self.min_gap)
            result.append(p)
            lo = p + self.min_gap

        return result

# GC implementation (algorithm 2)

class GreedyChoosingSchedule:
    """
    this is a discrete-time adaptive schedule via greedy selection of the time points. 

    At every gc iteration, the interior discretizations points t_k moves to the 
    candidate in the abailable pool that minimizes this:
        e(t_{k-1}, t_k) + E(t_k, t_{k+1}), while holding all the other points fixed.

    Parameters:

    available_points: list of float
        All time points the model was trained on (embedding available points).
        Must include 0.0 and 1.0.

    init_schedule: list of float
        Initial N-point schedule chosen from available_points.

    eps_fn: callable float -> float
        Flow matching error at t (only evaluated at available_points).

    L_fn: callable, optional
    C: float
    n_iters: int
    max_shift: float or None
        If set, restricts candidates to within +/- max_shift of the current
        point (this prevents large jumps caused by DAE/FME gap at small t).
    """

    def __init__(
        self,
        available_points: List[float],
        init_schedule: List[float],
        eps_fn: Callable[[float], float],
        L_fn: Callable[[float], float] = L_lipschitz,
        C: float = 10.0,
        n_iters: int = 5,
        max_shift: Optional[float] = None,
    ):
        self.available = sorted(available_points)
        self.eps_fn = eps_fn
        self.L_fn = L_fn
        self.C = C
        self.n_iters = n_iters
        self.max_shift = max_shift

        self._schedule = sorted(init_schedule)
        assert self._schedule[0]  == self.available[0],  "Schedule must start at available[0]"
        assert self._schedule[-1] == self.available[-1], "Schedule must end at available[-1]"

    @property
    def schedule(self) -> List[float]:
        return list(self._schedule)
    
    def _candidates_for(self, k: int) -> List[float]:
        """
        valid candidates for position k: have to lie strictly between t_{k-1}
        and t_{k+1}, also within available_points. Can aslo be constrained by max_shift
        """

        t_prev = self.schedule[k - 1]
        t_next = self._schedule[k + 1]
        t_cur = self._schedule[k]

        cands = [
            t for t in self.available
            if t_prev < t < t_next

        ]
        if self.max_shift is not None:
            cands = [
                t for t in cands
                if abs(t - t_cur) <= self.max_shift
            ]

        return cands
    
    def _local_EB(self, k: int, t: float) -> float:
        t_prev = self._schedule[k - 1]
        t_next = self._schedule[k + 1]
        return (
            compute_E(t_prev, t, self.eps_fn, self.L_fn, self.C)
          + compute_E(t, t_next, self.eps_fn, self.L_fn, self.C)
        )
    
    def run(self, verbose: bool = True) -> Tuple[List[float], List[float]]:
        """
        run the greedy choosing iterations. 

        Returns:

        final_schedule: sorted list of the N time points
        eb_history: EB at start of each outer iteration
        """

        eb_history = [compute_EB(self._schedule, self.eps_fn, self.L_fn, self.C)]

        for iteration in range(self.n_iters):
            n_moved = 0
            # skip the endpoints
            for k in range(1, len(self._schedule) - 1): 
                cands = self._candidates_for(k)
                if not cands:
                    continue
                best_t = min(cands, key=lambda t: self._local_EB(k, t))
                if best_t != self._schedule[k]:
                    self._schedule[k] = best_t
                    n_moved += 1
            
            eb = compute_EB(self._schedule, self.eps_fn, self.L_fn, self.C)
            eb_history.append(eb)

            if verbose:
                print(f"  GC iter {iteration+1} | EB = {eb:.6f} | moved {n_moved} points")

        return self.schedule, eb_history


# schedule builder / utilities.
def uniform_schedule(N: int, t0: float = 0.0, t1: float = 1.0) -> List[float]:
    # linearly spaced schedule with N steps (N+1 points).
    return [t0 + (t1 - t0) * i / N for i in range(N + 1)]

def to_tensor(schedule: List[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(schedule, dtype=torch.float32, device=device)