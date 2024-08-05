import numpy as np
import matplotlib.pyplot as plt

from coin import COIN


def main():
    coin_model = COIN() # use default parameters for now
    coin_model.perturbations = np.concatenate([
        np.zeros((50, )), 
        np.ones((125, )), 
        -np.ones((15, )), 
        np.ones((150, )) * np.nan, 
    ])
    
    output = coin_model.simulate_coin()
    
    plt.plot(output["runs"][0]["state_feedback"], "b.", label="state feedback")
    plt.plot(output["runs"][0]["motor_output"], "r", label="motor output")
    plt.legend()
    plt.savefig("figures/temp_test.png")
    
    return output


if __name__=="__main__":
    output = main()
