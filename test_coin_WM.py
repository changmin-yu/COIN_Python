import numpy as np
import matplotlib.pyplot as plt

from coin import COIN


def main():
    retention_values = np.linspace(0.8, 1, 500, endpoint=True)
    drift_values = np.linspace(-0.1, 0.1, 500, endpoint=True)
    state_values = np.linspace(-1.5, 1.5, 500, endpoint=True)
    bias_values = np.linspace(-1.5, 1.5, 500, endpoint=True)
    state_feedback_values = np.linspace(-1.5, 1.5, 500, endpoint=True)
    
    coin = COIN(
        retention_values=retention_values,
        drift_values=drift_values,
        state_values=state_values,
        bias_values=bias_values,
        state_feedback_values=state_feedback_values,
        store=["state_feedback", "motor_output"],
        sample_crf_stirling=False,  # use default parameters for now
        simple_sampling=True, 
        max_cores=4, 
    )
    coin.perturbations = np.concatenate([
        np.zeros((192, )), 
        np.ones((384, )), 
        -np.ones((20, )), 
        np.ones((192, )) * np.nan, 
    ])
    
    coin.runs = 10
    coin.plot_state_given_context = True
    coin.plot_predicted_probabilities = True
    coin.plot_state = True
    
    # deal with this later
    # coin.stationary_trials = 597
    
    output = coin.simulate_coin()

    return output


if __name__ == "__main__":
    output = main()
    