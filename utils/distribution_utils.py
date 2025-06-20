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
from typing import Union

from ..utils.general_utils import (
    per_slice_cholesky, 
    per_slice_multiply, 
)

from scipy.stats import truncnorm
from scipy.special import erfc, erfcinv


import numpy as np
from scipy.stats import beta

def random_binomial(p: Union[float, np.ndarray], n: Union[int, np.ndarray]):
    """
    Sample from a Binomial distribution with parameters n and p.
    
    Parameters:
    -----------
    p : float or ndarray
        Probability of success (must be between 0 and 1)
        Can be a scalar, 1D array, or n-dimensional array
    n : int or ndarray
        Number of trials (must be non-negative)
        Can be a scalar, 1D array, or n-dimensional array
        
    Returns:
    --------
    int or ndarray
        Random sample(s) from Binomial(n, p)
        Returns array with same shape as input p and n
        
    Notes:
    ------
    Uses three different methods depending on n and p:
    1. For small n (< 15) or when p > 0.5 and n < 15: Coin flip method (O(n) time)
    2. For small n*p (< 10): Waiting time method (O(np) time)
    3. For other cases: Recursive method based on beta distribution (O(log(log(n))) recursive calls)
    """
    # Convert scalar inputs to arrays
    is_scalar = np.isscalar(p) and np.isscalar(n)
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=int)
    
    # Ensure p and n have the same shape
    if p.shape != n.shape:
        raise ValueError("p and n must have the same shape")
    
    original_shape = p.shape
    
    # Flatten arrays for processing
    p_flat = p.ravel()
    n_flat = n.ravel()
    
    # Handle edge cases
    mask = np.isnan(p_flat) | (p_flat < 0) | (p_flat > 1) | (n_flat < 0)
    result = np.full_like(n_flat, 0, dtype=float)
    
    # Process valid inputs
    valid_mask = ~mask
    if not np.any(valid_mask):
        return result.reshape(original_shape)[0] if is_scalar else result.reshape(original_shape)
    
    # Initialize result array for valid inputs
    valid_indices = np.where(valid_mask)[0]
    result_flat = np.zeros_like(n_flat, dtype=float)
    
    # Process each valid input
    for i, idx in enumerate(valid_indices):
        n_val = n_flat[idx]
        p_val = p_flat[idx]
        
        # Special cases
        if n_val == 0:
            result_flat[idx] = 0
            continue
            
        if p_val == 0:
            result_flat[idx] = 0
            continue
            
        if p_val == 1:
            result_flat[idx] = n_val
            continue
            
        # Method 1: Coin flip method for small n or when p > 0.5 and n < 15
        if n_val < 15 or (p_val > 0.5 and n_val < 15):
            result_flat[idx] = np.sum(np.random.random(n_val) < p_val)
            continue
            
        # Method 2: Waiting time method for small n*p
        if n_val * p_val < 10:
            q = -np.log(1 - p_val)
            r = n_val
            e = -np.log(np.random.random())
            s = e/r
            while s <= q:
                r -= 1
                if r == 0:
                    break
                e = -np.log(np.random.random())
                s += e/r
            result_flat[idx] = n_val - r
            continue
            
        # Method 3: Recursive method using beta distribution
        i = int(p_val * (n_val + 1))
        b = beta.rvs(i, n_val + 1 - i)
        if b <= p_val:
            result_flat[idx] = i + random_binomial((p_val - b)/(1 - b), n_val - i)
        else:
            result_flat[idx] = i - 1 - random_binomial((b - p_val)/b, i - 1)
    
    # Reshape result to match input shape
    result = result_flat.reshape(original_shape)
    return result[0] if is_scalar else result


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


def random_gamma_simple(a: Union[np.ndarray, float]):
    """
    Sample from a Gamma distribution with shape parameter a and scale parameter 1.
    
    Parameters:
    -----------
    a : float or ndarray
        Shape parameter(s) of the gamma distribution (must be > 0)
        Can be a scalar, 1D array, or n-dimensional array
        
    Returns:
    --------
    float or ndarray
        Random sample(s) from Gamma(a, 1)
        Returns array with same shape as input a
        
    Notes:
    ------
    Uses Marsaglia and Tsang's method (2000) for a >= 1
    For a < 1, uses Marsaglia's (1961) method to boost the sample
    
    Algorithm from:
    G. Marsaglia and W.W. Tsang, A simple method for generating gamma
    variables, ACM Transactions on Mathematical Software, Vol. 26, No. 3,
    Pages 363-372, September, 2000.
    """
    # Convert scalar input to array
    is_scalar = np.isscalar(a)
    a = np.asarray(a, dtype=float)
    original_shape = a.shape
    
    # Flatten array for processing
    a_flat = a.ravel()
    
    # Handle edge cases
    mask = np.isnan(a_flat) | (a_flat <= 0)
    result = np.full_like(a_flat, 0)
    
    # Process valid inputs
    valid_mask = ~mask
    if not np.any(valid_mask):
        return result.reshape(original_shape)[0] if is_scalar else result.reshape(original_shape)
        
    # For a < 1, use Marsaglia's (1961) method: gam(a) = gam(a+1)*U^(1/a)
    boost = np.ones_like(a_flat)
    small_a_mask = valid_mask & (a_flat < 1)
    if np.any(small_a_mask):
        boost[small_a_mask] = np.exp(np.log(np.random.random(np.sum(small_a_mask))) / a_flat[small_a_mask])
        a_flat[small_a_mask] += 1
        
    # Marsaglia and Tsang's method for a >= 1
    d = a_flat - 1.0/3
    c = 1.0 / np.sqrt(9*d)
    
    # Generate samples for all valid inputs
    valid_indices = np.where(valid_mask)[0]
    for idx in valid_indices:
        while True:
            # Generate normal random variable
            x = np.random.normal()
            v = 1 + c[idx]*x
            
            if v <= 0:
                continue
                
            v = v**3
            x2 = x*x
            u = np.random.random()
            
            if (u < 1 - 0.0331*x2*x2) or (np.log(u) < 0.5*x2 + d[idx]*(1 - v + np.log(v))):
                result[idx] = boost[idx] * d[idx] * v
                break
    
    # Reshape result to match input shape
    result = result.reshape(original_shape)
    return result[0] if is_scalar else result
                
                
def random_gamma_ND(R: np.ndarray):
    gamma_samples = np.zeros_like(R, dtype=float)
    
    l, w = R.shape
    
    for i in range(l):
        for j in range(w):
            gamma_samples[i, j] = random_gamma(R[i, j])
            
    return gamma_samples
    

def random_dirichlet(R: np.ndarray):
    # gamma_samples = random_gamma_ND(R)
    gamma_samples = random_gamma_simple(R)
    return gamma_samples / np.sum(gamma_samples, axis=0, keepdims=True)


def stationary_distribution(transmat: np.ndarray):
    # statioanry distribution of a time-homogeneous Markov process
    
    c = transmat.shape[0] # number of contexts
    
    A = transmat.T - np.eye(c)
    b = np.zeros((c, ))
    
    # add normalisation constraint to ensure proper probability distribution
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
    tolerance = 2
    inds = (np.abs(u - l) > tolerance)
    x = l.copy()
    
    # case: abs(u - l) > tolerance, use acccept-reject from randn
    if np.any(inds):
        tl = l[inds]
        tu = u[inds]
        x[inds] = trnd(tl, tu)
    
    # case: abs(u - l) < tolerance, use cumulative normal inverse-transform
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