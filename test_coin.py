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
import matplotlib.pyplot as plt

from coin import COIN


def main():
    # coin_model = COIN(sample_crf_stirling=False) # use default parameters for now
    # coin_model.perturbations = np.concatenate([
    #     np.zeros((50, )), 
    #     np.ones((125, )), 
    #     -np.ones((15, )), 
    #     np.ones((150, )) * np.nan, 
    # ])
    
    # output = coin_model.simulate_coin()
    
    retention_values = np.linspace(0.8, 1, 500, endpoint=True)
    drift_values = np.linspace(-0.1, 0.1, 500, endpoint=True)
    state_values = np.linspace(-1.5, 1.5, 500, endpoint=True)
    bias_values = np.linspace(-1.5, 1.5, 500, endpoint=True)
    state_feedback_values = np.linspace(-1.5, 1.5, 500, endpoint=True)

    store = [
        "state_feedback", "motor_output", "responsibilities", 
    ]

    coin_model = COIN(
        retention_values=retention_values, 
        drift_values=drift_values, 
        state_values=state_values, 
        bias_values=bias_values, 
        state_feedback_values=state_feedback_values, 
        store=store, 
        sample_crf_stirling=True, 
    ) # use default parameters for now
    coin_model.perturbations = np.concatenate([
        np.zeros((50, )), 
        np.ones((125, )), 
        -np.ones((15, )), 
        np.ones((150, )) * np.nan, 
    ])

    output = coin_model.simulate_coin()
    
    plt.figure(figsize=(8, 5))
    plt.plot(output["runs"][0]["state_feedback"], "b.", label="state feedback")
    plt.plot(output["runs"][0]["motor_output"], "r", label="motor output")
    plt.legend()
    plt.savefig("figures/temp_test.png")
    
    return output


if __name__=="__main__":
    output = main()
