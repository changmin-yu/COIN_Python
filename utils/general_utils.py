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


from typing import List
import numpy as np


def sample_num_tables_CRF(
    weights: np.ndarray, 
    max_tables: np.ndarray, 
    max_n: int = 1, 
    all_stirling_number: List[np.ndarray] = [np.array([1])], 
    log_max_stirling_number: List[int] = [0], 
):
    """
    Sample the number of tables in restaurant i serving dish j
    
    params
        weights (np.ndarray): weights for each dish
        max_tables (np.ndarray): number of maximum number of tables within each restaurant
    """
    num_tables = np.zeros_like(max_tables, dtype=max_tables.dtype) # (max_contexts+1, max_contexts+1, particles)
    
    B, I, J = np.unique(max_tables.flatten(order="F"), return_index=True, return_inverse=True)
    
    log_weights = np.log(weights)
    
    for i in range(len(B)):
        max_table = B[i]
        
        if max_table > 0:
            table_list = np.arange(1, max_table+1)
            (
                stirling_num, 
                curr_log_max_stirling_number, 
                max_n, 
                all_stirling_number, 
                log_max_stirling_number, 
            ) = stirling_number(
                max_table, 
                max_n, 
                all_stirling_number, 
                log_max_stirling_number, 
            )
            for j in np.where(J == i)[0]:
                inds1, inds2, inds3 = np.unravel_index(j, max_tables.shape, order="F")
                c_likelihood = table_list * log_weights[inds1, inds2, inds3]
                c_likelihood = np.cumsum(stirling_num * np.exp(c_likelihood - np.max(c_likelihood)), axis=0)
                # TODO: verify if the following line is actually correct (by comparing with the matlab implementation)
                num_tables[inds1, inds2, inds3] = 1 + np.sum((np.random.rand() * c_likelihood[max_table-1]) > c_likelihood, axis=0)
    
    return num_tables, max_n, all_stirling_number, log_max_stirling_number


def stirling_number(
    n: int, 
    max_n: int = 1, 
    all_stirling_number: List[np.ndarray] = [np.array([1])], 
    log_max_stirling_number: List[int] = [0], 
) -> np.ndarray:
    # compute the unsigned stirling numbers of the first kind
    
    if n > max_n:
        for i in range(max_n, n):
            if len(all_stirling_number) > i:
                all_stirling_number[i] = np.concatenate([all_stirling_number[i-1] * i, np.array([0])], axis=0) + \
                    np.concatenate([np.array([0]), all_stirling_number[i-1]], axis=0)
            else:
                all_stirling_number.append(
                    np.concatenate([all_stirling_number[i-1] * i, np.array([0])], axis=0) + 
                    np.concatenate([np.array([0]), all_stirling_number[i-1]], axis=0)
                )
            max_curr = np.max(all_stirling_number[i])
            all_stirling_number[i] = all_stirling_number[i] / max_curr
            
            if len(log_max_stirling_number) > i:
                log_max_stirling_number[i] = log_max_stirling_number[i-1] + np.log(max_curr)
            else:
                log_max_stirling_number.append(log_max_stirling_number[i-1] + np.log(max_curr))

        max_n = n
    
    curr_stirling_number = all_stirling_number[n-1]
    curr_log_max_stirling_number = log_max_stirling_number[n-1]
    
    return (
        curr_stirling_number, 
        curr_log_max_stirling_number, 
        max_n, 
        all_stirling_number, 
        log_max_stirling_number, 
    )
    
    
def per_slice_invert(L: np.ndarray):
    L_determinant = L[0, 0] * L[1, 1] - L[1, 0] * L[0, 1]
    L_inverse = np.array([
        np.array([L[1, 1] * 1, L[0, 1] * (-1)]), 
        np.array([L[1, 0] * (-1), L[0, 0] * 1]), 
    ]) / L_determinant
    
    return L_inverse


def per_slice_multiply(A: np.ndarray, B: np.ndarray):
    C = np.sum(A * np.transpose(B, (2, 0, 1)), axis=1)
    
    return C


def per_slice_cholesky(X: np.ndarray):
    L = np.zeros_like(X, dtype=X.dtype)
    L[0, 0] = np.sqrt(X[0, 0])
    L[1, 0] = X[1, 0] / L[0, 0]
    L[1, 1] = np.sqrt(X[1, 1] - np.square(X[1, 0]))
    
    return L


def log_sum_exp(log_probs: np.ndarray, axis: int = 0):
    m = np.max(log_probs, axis=axis)
    l = m + np.log(np.sum(np.exp(log_probs - m), axis=axis))
    
    return l


def systematic_resampling(weights: np.ndarray):
    n = weights.size
    Q = np.cumsum(weights)
    y = np.linspace(0, 1-1/n, n, endpoint=True) + np.random.rand() / n
    p = np.zeros((n, ), dtype=int)
    i, j = 0, 0
    while i < n and j < n:
        while Q[j] < y[i]:
            j += 1
        p[i] = j
        i += 1
    
    return p


if __name__=="__main__":
    n = 10
    
    # checked for n is a scalar
    ss, max_n, all_stirling_number, log_max_stirling_number = stirling_number(n)
    
    pass