"""
Copyright (c) 2024 Changmin Yu

This file is part of COIN_Python.

COIN_Python is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

COIN_Python is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with COIN_Python. If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np

from utils.general_utils import (
    per_slice_cholesky, 
    per_slice_multiply, 
)

from scipy.stats import truncnorm
from scipy.special import erfc, erfcinv


def random_gamma(r: float):
    # sample gamma random variable
    
    if r <= 0:
        return 0.0
    
    if r == 1.0:
        return -np.log(np.random.rand())
    
    if r < 1.0:
        # Johnks generator
        c = 1 / r
        d = 1 / (1 - r)
        
        while True:
            x = np.random.rand() ** c
            y = x + np.random.rand() ** d
            
            if y <= 1.0:
                g = -np.log(np.random.rand()) * x / y
                return g
    
    if r > 1.0:
        # Best's algorithm
        b = r - 1.0
        c = 3.0 * r - 0.75
        
        while True:
            u = np.random.rand()
            v = np.random.rand()
            w = u * (1.0 - u)
            y = np.sqrt(c / w) * (u - 0.5)
            x = b + y
            
            if x >= 0.0:
                z = 64.0 * (w ** 3) * (v ** 2)
                if (z <= (1.0 - 2.0 * (y ** 2) / x)) or (np.log(z) <= 2.0 * (b * np.log(x / b) - y)):
                    g = x
                    return g
                
                
def random_gamma_ND(R: np.ndarray):
    gamma_samples = np.zeros_like(R, dtype=float)
    
    l, w = R.shape
    
    for i in range(l):
        for j in range(w):
            gamma_samples[i, j] = random_gamma(R[i, j])
            
    return gamma_samples
    

def random_dirichlet(R: np.ndarray):
    gamma_samples = random_gamma_ND(R)
    return gamma_samples / np.sum(gamma_samples, axis=0, keepdims=True)


def stationary_distribution(transmat: np.ndarray):
    # statioanry distribution of a time-homogeneous Markov process
    
    c = transmat.shape[0] # number of contexts
    
    A = transmat.T - np.eye(c)
    b = np.zeros((c, ))
    
    A = np.concatenate([A, np.ones((1, c))], axis=0)
    b = np.concatenate([b, np.array([1.])], axis=0)
    
    x = np.linalg.solve(A, b)
    
    x[x < 0] = 0
    p = x / np.sum(x)

    return p


def random_truncated_bivariate_normal(mu: np.ndarray, cov: np.ndarray):
    cov_cholesky = per_slice_cholesky(cov)
    
    lb, ub = 0.0, 1.0 # truncation bounds
    
    # equivalent truncation bounds for the standard normal distribution
    lb_normal = np.array([
        (lb - mu[0, :]) / cov_cholesky[0, 0], 
        -np.inf * np.ones((cov.shape[-1])), 
    ])
    ub_normal = np.array([
        (ub - mu[0, :]) / cov_cholesky[0, 0], 
        np.inf * np.ones((cov.shape[-1])), 
    ])
    
    # transform samples from the truncated standard normal distribution to 
    # samples from the desired truncated normal distribution
    # x = mu + per_slice_multiply(cov_cholesky, np.reshape(truncnorm.rvs(lb_normal, ub_normal), [2, cov.shape[2]])[..., None]) # TODO: use numpy/scipy truncated normal generator
    # use self-defined truncated-normal random generator
    samples = trandn(lb_normal, ub_normal)
    x = mu + per_slice_multiply(cov_cholesky, np.reshape(samples, [2, cov.shape[2]])[..., None])
    
    return x


def random_univariate_normal(
    mu: np.ndarray, 
    sigma: np.ndarray, 
    num_particles: int, 
    max_contexts: int, 
):
    x = mu[:, None] + np.sqrt(sigma[:, None]) * np.random.randn(max_contexts+1, num_particles)
    
    return x


def trandn(l: np.ndarray, u: np.ndarray):
    """
    Truncated normal generator
    """
    
    l = l.flatten()
    u = u.flatten()
    # l = l.flatten()
    # u = u.flatten()
    
    if len(l) != len(u):
        raise ValueError("Truncation limits have to be vectors of the same length!")

    x = np.ones_like(l) * np.nan
    a = 0.66 # threshold for switching between methods
    
    inds_1 = (l > a)
    # case 1: a < l < u
    if np.any(inds_1):
        tl = l[inds_1]
        tu = u[inds_1]
        x[inds_1] = ntail(tl, tu)
    
    # case 2: l < u < -a
    inds_2 = (u < -a)
    if np.any(inds_2):
        tl = -u[inds_2]
        tu = -l[inds_2]
        x[inds_2] = -ntail(tl, tu)
    
    # case 3: otherwise we use inverse transform or accept-reject
    inds = ~(inds_1 | inds_2)
    if np.any(inds):
        tl = l[inds]
        tu = u[inds]
        x[inds] = tn(tl, tu)
    
    return x


def ntail(l: np.ndarray, u: np.ndarray):
    # sample from standard normal distribution, truncated over the region [l, u]
    # then use acceptance-rejection from Rayleigh distribution
    
    c = np.square(l) / 2
    n = len(l)
    f = np.expm1(c - np.square(u) / 2)
    x = c - np.log(1 + np.random.rand(n) * f) # sample using Rayleigh
    # keep list of rejected
    inds = np.where((np.square(np.random.rand(n)) * x) > c)[0]
    d = len(inds)
    while d > 0:
        cy = c[inds]
        y = cy - np.log(1 + np.random.rand(d) * f[inds])
        accepted_inds = (np.square(np.random.rand(d)) * y) < cy
        x[inds[accepted_inds]] = y[accepted_inds]
        inds = inds[~accepted_inds]
        d = len(inds)
    
    x = np.sqrt(2 * x)
    return x


def tn(l: np.ndarray, u: np.ndarray):
    # sample from standard normal distribution, truncated over the region [l, u]
    # where -a < l < u < a
    tolerance = 2
    inds = (np.abs(u - l) > tolerance)
    x = l.copy()
    
    # case: abs(u - l) > tolerance, use acccept-reject from randn
    if np.any(inds):
        tl = l[inds]
        tu = u[inds]
        x[inds] = trnd(tl, tu)
    
    # case: abs(u - l) < tolerance, use inverse-transform
    inds = ~inds
    if np.any(inds):
        tl = l[inds]
        tu = u[inds]
        pl = erfc(tl / np.sqrt(2)) / 2
        pu = erfc(tu / np.sqrt(2)) / 2
        x[inds] = np.sqrt(2) * erfcinv(2 * (pl - (pl - pu) * np.random.rand(len(tl))))
    
    return x


def trnd(l: np.ndarray, u: np.ndarray):
    x = np.random.randn(len(l))
    inds = np.where(((x < l) | (x > u)))[0]
    d = len(inds)
    
    while d > 0:
        ly = l[inds]
        uy = u[inds]
        y = np.random.randn(len(ly))
        accepted_inds = np.logical_and(y > ly, y < uy)
        x[inds[accepted_inds]] = y[accepted_inds]
        inds = inds[~accepted_inds]
        d = len(inds)
    
    return x