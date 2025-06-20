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


from typing import Optional, List, Dict, Any

import os

import numpy as np

from scipy.stats import norm

from itertools import permutations
from math import factorial

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from tqdm import trange

import multiprocessing
import functools

from .utils.general_utils import (
    sample_num_tables_CRF, 
    per_slice_invert, 
    per_slice_multiply, 
    log_sum_exp, 
    systematic_resampling, 
    check_cue_labels,
    randnumtable_array, 
)
from .utils.distribution_utils import (
    random_dirichlet, 
    stationary_distribution, 
    random_truncated_bivariate_normal, 
    random_univariate_normal, 
    random_binomial, 
)


class COIN:
    """
    params:
        core parameters
            sigma_process_noise (float): standard deviation of process noise;
            sigma_sensory_noise (float): standard deviation of sensory noise;
            sigma_motor_noise (float): standard deviation of motor noise
            prior_mean_retention (float): prior mean of retention
            prior_precision_retention (float): prior precision of retention
            prior_precision_drift (float): prior precision of drift
            gamma_context (float): gamma hyperparameter of the CRF for the context transitions
            alpha_context (float): alpha hyperparameter of the CRF for the context transitions
            rho_context (float): rho (normalised self-transition) hyperparameter of the CRF for the context transitions
        parameters if cues are present
            gamma_cue (float): gamma hyperparmeter of the CRF for the cue emissions
            alpha_cue (float): alpha hyperparmeter of the CRF for the cue emissions
        parameters if inferring bias
            infer_bias (bool): infer the measurement bias
            prior_precision_bias (float): precision of the prior of measurement bias
        paradigm
            perturbations (np.ndarray): vector of perturbations (use nan on channel trials)
            cues (np.ndarray): vector of sensory cues
            stationary_trials (List[int]): trials on which to set the predicted probabilities to the stationary probabilities (e.g., following a working memory task)
        runs
            runs (int): number of runs, each conditioned on a different state feedback sequence
        parallel processing of runs
            parallel_processing (bool): to use or not parallel processing over runs
            max_cores (int): maximum number of CPU cores available (1 implements serial processing of runs)
        model implementation
            particles (int): number of particles
            max_contexts (int): maximum number of contexts that can be instantiated
        measured adaptation data
            adaptation (np.ndarray): vector of adaptation data (use nan on trials where adaptation was not measured)
        store
            store (List[str]): variables to store
        evaluation inputs
            retention_values (np.ndarray): values at which to evaluate p(retention)
            drift_values (np.ndarray): values at which to evaluate p(drift)
            state_values (np.ndarray): values at which to evaluate p(state)
            bias_values (np.ndarray): values at which to evaluate p(bias)
            state_feedback_values (np.ndarray): values at which to evaluate p(state_feedback)
        plotting flags
            plot_state_given_context (bool): plot state|context distrbution (predicted state distribution for each context)
            plot_predicted_probabilities (bool): plot predicted probabilities
            plot_responsibilities (bool): plot responsibilities
            plot_stationary_probabilities (bool): plot stationary probabilities
            plot_retention_given_context (bool): plot retention|context distribution
            plot_drift_given_context (bool): plot drift|context distribution
            plot_bias_given_context (bool): plot bias_given_context distribution
            plot_global_transition_probabilities (bool): plot global transition probabilities
            plot_local_transition_probabilities (bool): plot local transition probabilities
            plot_global_cue_probabilities (bool): plot global cue probabilities
            plot_local_cue_probabilities (bool): plot local cue probabilities
            plot_state (bool): plot state (overall predicted state distribution)
            plot_average_state (bool): plot average state (mean of overall predicted state distribution)
            plot_bias (bool):  plot bias distribution (average bias distribution across contexts)
            plot_average_bias (bool): plot average bias (mean of the average bias distribution across contexts)
            plot_state_feedback (bool): plot predicted state feedback distribution (average state feedback distribution across contexts)
            plot_explicit_component (bool): plot explicit component of learning
            plot_implicit_component (bool): plot implicit component learning
            plot_Kalman_gain_given_cstar1 (bool): plot Kalman_gain|context with highest responsibility on current trial (cstar1)
            plot_predicted_probability_cstar1 (bool): plot predicted probability of contextwith highest responsibility on current trial (cstar1)
            plot_state_given_cstar1 (bool): plot state|context with highest responsibility on current trial (cstar1)
            plot_Kalman_gain_given_cstar2 (bool): plot Kalman_gain|context with highest predicted probability on next trial (cstar2)
            plot_state_given_cstar2 (bool): plot state|context with highest predicted probability on next trial (cstar2)
            plot_predicted_probability_cstar3 (bool): plot predicted probability of context with highest predicted probability on current trial (cstar3)
            plot_state_given_cstar3 (bool): plot state|context with highest predicted probability on current trial (cstar3)
        figure directory
            fig_dir (str): base directory to save figures
    """
    
    def __init__(
        self, 
        sigma_process_noise: float = 0.0089, 
        sigma_sensory_noise: float = 0.03, 
        sigma_motor_noise: float = 0.0182, 
        prior_mean_retention: float = 0.9425, 
        prior_precision_retention: float = 837.1 ** 2, 
        prior_precision_drift: float = 1.2227e3 ** 2, 
        gamma_context: float = 0.1, 
        alpha_context: float = 8.955, 
        rho_context: float = 0.2501, 
        gamma_cue: float=  0.1, 
        alpha_cue: float = 25, 
        infer_bias: bool = False, 
        prior_precision_bias: float = 70 ** 2, 
        # paradigm
        perturbations: Optional[np.ndarray] = None, 
        cues: Optional[np.ndarray] = None, 
        stationary_trials: List[int] = [], 
        # runs
        runs: int = 1, 
        # parllel processing
        max_cores: int = 1, 
        # model implementation
        particles: int = 100, 
        max_contexts: int = 10, 
        # measured adaptation
        adaptation: Optional[np.ndarray] = None, 
        # store
        store: List[str] = ["state_feedback", "motor_output"], 
        # evaluation
        retention_values: Optional[np.ndarray] = None, 
        drift_values: Optional[np.ndarray] = None, 
        state_values: Optional[np.ndarray] = None, 
        bias_values: Optional[np.ndarray] = None, 
        state_feedback_values: Optional[np.ndarray] = None, 
        # plot flags
        plot_state_given_context: bool = False, 
        plot_predicted_probabilities: bool = False, 
        plot_responsibilities: bool = False, 
        plot_stationary_probabilities: bool = False, 
        plot_retention_given_context: bool = False, 
        plot_drift_given_context: bool = False, 
        plot_bias_given_context: bool = False, 
        plot_global_transition_probabilities: bool = False, 
        plot_local_transition_probabilities: bool = False, 
        plot_global_cue_probabilities: bool = False, 
        plot_local_cue_probabilities: bool = False, 
        plot_state: bool = False, 
        plot_average_state: bool = False, 
        plot_bias: bool = False, 
        plot_average_bias: bool = False, 
        plot_state_feedback: bool = False, 
        plot_explicit_component: bool = False, 
        plot_implicit_component: bool = False, 
        plot_Kalman_gain_given_cstar1: bool = False, 
        plot_predicted_probability_cstar1: bool = False, 
        plot_state_given_cstar1: bool = False, 
        plot_Kalman_gain_given_cstar2: bool = False, 
        plot_state_given_cstar2: bool = False, 
        plot_predicted_probability_cstar3: bool = False, 
        plot_state_given_cstar3: bool = False, 
        # figure directory
        fig_dir: Optional[str] = None, 
        # misc
        sample_crf_stirling: bool = True, 
    ):
        self.sigma_process_noise = sigma_process_noise # sigma_q (eq. 3)
        self.sigma_sensory_noise = sigma_sensory_noise # sigma_r (eq. 5)
        self.sigma_motor_noise = sigma_motor_noise
        self.prior_mean_retention = prior_mean_retention # mu_a (eq. 10)
        self.prior_precision_retention = prior_precision_retention # 1/sigma_a^2 (eq. 10)
        self.prior_precision_drift = prior_precision_drift # 1 / sigma_d^2 (eq. 10)
        self.gamma_context = gamma_context # gamma (eq. 7)
        self.alpha_context = alpha_context # alpha (eq. 8)
        self.rho_context = rho_context # self-transition bias (eq. S10)
        self.gamma_cue = gamma_cue # gamma_e (eq. 9)
        self.alpha_cue = alpha_cue # alpha_e (eq. 9)
        self.infer_bias = infer_bias
        self.prior_precision_bias = prior_precision_bias # 1/sigma_b^2 (eq. 20)
        
        self.perturbations = perturbations
        self.cues = cues
        self.stationary_trials = stationary_trials
        
        self.runs = runs
        
        self.max_cores = max_cores
        if self.max_cores == 1: self.parallel_processing = False
        
        self.particles = particles # number of particles
        self.max_contexts = max_contexts # maximum number of contexts that can be instantiated
        
        self.adaptation = adaptation
        
        self.store = store
        
        self.plot_state_given_context = plot_state_given_context
        self.plot_predicted_probabilities = plot_predicted_probabilities
        self.plot_responsibilities = plot_responsibilities
        self.plot_stationary_probabilities = plot_stationary_probabilities
        self.plot_retention_given_context = plot_retention_given_context
        self.plot_drift_given_context = plot_drift_given_context
        self.plot_bias_given_context = plot_bias_given_context
        self.plot_global_transition_probabilities = plot_global_transition_probabilities
        self.plot_local_transition_probabilities = plot_local_transition_probabilities
        self.plot_global_cue_probabilities = plot_global_cue_probabilities
        self.plot_local_cue_probabilities = plot_local_cue_probabilities
        self.plot_state = plot_state
        self.plot_average_state = plot_average_state
        self.plot_bias = plot_bias
        self.plot_average_bias = plot_average_bias
        self.plot_state_feedback = plot_state_feedback
        self.plot_explicit_component = plot_explicit_component
        self.plot_implicit_component = plot_implicit_component
        self.plot_Kalman_gain_given_cstar1 = plot_Kalman_gain_given_cstar1
        self.plot_predicted_probability_cstar1 = plot_predicted_probability_cstar1
        self.plot_state_given_cstar1 = plot_state_given_cstar1
        self.plot_Kalman_gain_given_cstar2 = plot_Kalman_gain_given_cstar2
        self.plot_state_given_cstar2 = plot_state_given_cstar2
        self.plot_predicted_probability_cstar3 = plot_predicted_probability_cstar3
        self.plot_state_given_cstar3 = plot_state_given_cstar3
        
        # for plotting
        self.retention_values = retention_values
        self.drift_values = drift_values
        self.state_values = state_values
        self.bias_values = bias_values
        self.state_feedback_values = state_feedback_values
        
        if fig_dir is None:
            fig_dir = "figures/"
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        
        self.sample_crf_stirling = sample_crf_stirling
        
    def simulate_coin(self):
        if self.cues is not None:
            self.cues = check_cue_labels(self.cues, self.perturbations)
        
        assert self.perturbations is not None, "perturbations unfound!"
        
        temp = []
        
        # set the store property based on the plot flags
        self.set_store_property_for_plots()
        
        # number of trials
        num_trials = len(self.perturbations)
        
        if (self.adaptation is None) or (len(self.adaptation) == 0):
            trials = np.arange(num_trials)
            
            print("Simulating the COIN model")
            
            # with multiprocessing.Pool(processes=self.max_cores) as pool:
            #     results = pool.map(parallel_coin_generative_main_loop, range(self.runs))
            # temp = results
            with trange(self.runs, dynamic_ncols=True) as pbar:
                for n in pbar:
                    coin_state = self.coin_generative_main_loop(trials)
                    temp.append(coin_state["stored"])

            w = np.ones(self.runs) / self.runs
            
        else:
            if len(self.adaptation) != len(self.perturbations):
                raise ValueError("Property ``adaptation'' should be a vector with one element per trial (use nan on trials where adaptation is not measured).")

            coin_state_in = {i: None for i in range(self.runs)}
            coin_state_out = {i: None for i in range(self.runs)}
            
            w = np.ones((self.runs, )) / self.runs
            
            # effective sample size threshold for resampling
            ESS_threshold = 0.5 * self.runs
            
            # trials on which adaptation is measured
            adaptation_trials = np.where(~np.isnan(self.adaptation))[0]
            
            # simulate trials in between trials on which adaptation was measured
            for i in range(len(adaptation_trials)):
                print(i)
                if i == 0:
                    trials = np.arange(adaptation_trials[i]+1)
                    print(f"Simulating the COIN model from trial 0 to trial {adaptation_trials[i]}")
                else:
                    trials = np.arange(adaptation_trials[i-1]+1, adaptation_trials[i]+1)
                    print(f"Simulating the COIN model from trial {adaptation_trials[i-1]+1} to trial {adaptation_trials[i]}")
                    
                for n in range(self.runs):
                    if i == 0:
                        coin_state_out[i] = self.coin_generative_main_loop(trials)
                    else:
                        coin_state_out[i] = self.coin_generative_main_loop(trials, coin_state_in[i])
                
                # calculate the log-likelihood
                log_likelihood = np.zeros((self.runs, ))
                for n in range(self.runs):
                    model_error = coin_state_out[n]["stored"]["motor_output"][adaptation_trials[i]] - self.adaptation[adaptation_trials[i]]
                    log_likelihood[n] = -np.log(2 * np.pi * np.square(self.sigma_motor_noise)) - \
                        np.square(model_error / self.sigma_motor_noise) / 2
                    
                # update weights and normalise
                l_w = log_likelihood + np.log(w)
                l_w = l_w - log_sum_exp(l_w, axis=1)
                w = np.exp(l_w)
                
                # calculate effect sample size
                ESS = 1 / np.sum(np.square(w))
                
                # if the effective sample size falls below threshold, resample
                if ESS < ESS_threshold:
                    print(f"Effect sample size = {ESS:.1f}, run resampling")
                    inds_resampled = systematic_resampling(w)
                    for n in range(self.runs):
                        coin_state_in[n] = coin_state_out[inds_resampled[n]]
                    w = np.ones((self.runs, )) / self.runs
                else:
                    print(f"Effective sample size = {ESS:.1f}")
                    coin_state_in = coin_state_out
                
            if adaptation_trials[-1] == (num_trials - 1):
                for n in range(self.runs):
                    temp.append(coin_state_in[n]["stored"])
            elif adaptation_trials[-1] < (num_trials - 1):
                print(f"Simulating COIN model from trial {adaptation_trials[-1]+1} to trial {num_trials}")
                
                trials = np.arange(adaptation_trials[-1]+1, num_trials)
                
                for n in range(self.runs):
                    temp.append(self.coin_generative_main_loop(trials, coin_state_in[n])["stored"])
        
        S = {}
        S["runs"] = {}
        
        for n in range(self.runs):
            S["runs"][n] = temp[n]
        S["weights"] = w
        S["properties"] = self
        
        properties = list(self.__dict__.keys())
        for i in range(len(properties)):
            if "plot" in properties[i]:
                if self.__dict__[properties[i]]:
                    S["plots"] = self.plot_coin(S)
                    break
        
        return S    
        
    def coin_generative_main_loop(self, trials: int, coin_state: Optional[Dict[str, Any]]=None):
        if trials[0] == 0:
            coin_state = self.initialise_coin()
        for trial in trials:
            coin_state["trial"] = trial
            coin_state = self.predict_context(coin_state)
            coin_state = self.predict_states(coin_state)
            coin_state = self.predict_state_feedback(coin_state)
            coin_state = self.resample_particles(coin_state)
            coin_state = self.sample_context(coin_state)
            coin_state = self.update_belief_about_states(coin_state)
            coin_state = self.sample_states(coin_state)
            coin_state = self.update_sufficient_statistics_for_parameters(coin_state)
            coin_state = self.sample_parameters(coin_state)
            coin_state = self.store_variables(coin_state)
        
        return coin_state
    
    def initialise_coin(self):
        coin_state = {}
        
        # number of trials
        coin_state["num_trials"] = len(self.perturbations)
        
        # is state feedback observed or not
        coin_state["feedback_observed"] = np.ones((len(self.perturbations), ))
        coin_state["feedback_observed"][np.isnan(self.perturbations)] = 0
        
        # self-transition bias
        coin_state["kappa"] = self.alpha_context * self.rho_context / (1 - self.rho_context) # (eq. S10)
        
        # observation noise standard deviation
        coin_state["sigma_observation_noise"] = np.sqrt(self.sigma_sensory_noise ** 2 + self.sigma_motor_noise ** 2)
        
        # matrix of context-dependent observation vectors
        coin_state["H"] = np.eye(self.max_contexts + 1)
        
        # current trial
        coin_state["trial"] = 0
        
        # number of contexts instantiated so far
        coin_state["C"] = np.zeros((self.particles, ), dtype=int)
        
        # context transition counts
        coin_state["n_context"] = np.zeros((self.max_contexts + 1, self.max_contexts + 1, self.particles), dtype=int)
        
        # sampled context
        coin_state["context"] = np.ones((self.particles, ), dtype=int)
        
        # do cues exist?
        if self.cues is None:
            coin_state["cues_exist"] = 0
        else:
            coin_state["cues_exist"] = 1
            
            # number of contextual cues observed so far
            coin_state["Q"] = 0
            
            # cue emission counts
            coin_state["n_cue"] = np.zeros((self.max_contexts + 1, np.max(self.cues) + 1, self.particles))
            
        # sufficient statistics for the parameters of the state dynamics function
        coin_state["dynamics_ss_1"] = np.zeros((self.max_contexts + 1, self.particles, 2))
        coin_state["dynamics_ss_2"] = np.zeros((self.max_contexts + 1, self.particles, 2, 2))
        
        # sufficient statistics for the parameters of the observation function
        coin_state["bias_ss_1"] = np.zeros((self.max_contexts + 1, self.particles))
        coin_state["bias_ss_2"] = np.zeros((self.max_contexts + 1, self.particles))
        
        # sample parameters from the prior
        coin_state = self.sample_parameters(coin_state)
        
        # mean and variance of state (stationary distribution)
        coin_state["state_filtered_mean"] = coin_state["drift"] / (1 - coin_state["retention"]) # (eq. 4)
        coin_state["state_filtered_var"] = (self.sigma_process_noise ** 2) / (1 - np.square(coin_state["retention"]))
        
        return coin_state
    
    def predict_context(self, coin_state: Dict[str, Any]):
        # re-initialise the context probabilities to their stationary values if it gets erased
        if coin_state["trial"] in self.stationary_trials:
            for p in range(self.particles):
                C = np.sum(coin_state["local_transition_matrix"][:, 0, p] > 0)
                transmat = coin_state["local_transition_matrix"][:C, :C, p]
                coin_state["prior_probabilities"][:C, p] = stationary_distribution(transmat)
        else:
            prior_probabilities = np.zeros((self.max_contexts+1, self.particles))
            
            inds_1 = np.tile(coin_state["context"][None], (self.max_contexts+1, 1)) - 1 # for indexing
            inds_2 = np.tile(np.arange(self.max_contexts+1)[None], (self.particles, 1)).T
            inds_3 = np.tile(np.arange(self.particles)[None], (self.max_contexts+1, 1))
            for i in range(self.max_contexts+1):
                for j in range(self.particles):
                    prior_probabilities[i, j] = coin_state["local_transition_matrix"][
                        inds_1[i, j], inds_2[i, j], inds_3[i, j], 
                    ]
            
            coin_state["prior_probabilities"] = prior_probabilities
            
            # TODO: verify if the following vectorised version works
            
            # inds_1 = np.tile(coin_state["context"][None], (self.max_contexts+1, 1)).ravel()
            # inds_2 = np.tile(np.arange(self.max_contexts+1)[:, None], (1, self.particles)).ravel()
            # inds_3 = np.tile(np.arange(self.particles)[None], (self.max_contexts+1, 1)).ravel()

            # coin_state["prior_probabilities"] = coin_state["local_transition_matrix"][inds_1, inds_2, inds_3].reshape(self.max_contexts+1, self.particles)

        if coin_state["cues_exist"]:
            cue_probabilities = np.zeros((self.max_contexts+1, self.particles))
            inds_1 = np.tile(np.arange(self.max_contexts+1)[None], (self.particles, 1)).T
            inds_2 = np.ones((self.max_contexts+1, self.particles), dtype=int) * self.cues[coin_state["trial"]-1]
            inds_3 = np.tile(np.arange(self.particles)[None], (self.max_contexts+1, 1))
            
            for i in range(self.max_contexts+1):
                for j in range(self.particles):
                    cue_probabilities[i, j] = coin_state["local_cue_matrix"][
                        inds_1[i, j], inds_2[i, j], inds_3[i, j], 
                    ]
            coin_state["cue_probabilities"] = cue_probabilities
            
            coin_state["predicted_probabilities"] = coin_state["prior_probabilities"] * coin_state["cue_probabilities"]
            coin_state["predicted_probabilities"] = coin_state["predicted_probabilities"] / np.sum(coin_state["predicted_probabilities"], axis=0, keepdims=True)
        
        else:
            coin_state["predicted_probabilities"] = coin_state["prior_probabilities"]
            
        if "Kalman_gain_given_cstar2" in self.store:
            if coin_state["trial"] > 0:
                max_inds = np.argmax(coin_state["predicted_probabilities"], axis=0)
                inds = np.arange(self.particles)
                
                assert len(max_inds) == self.particles
                
                kalman_gain = np.zeros((self.particles, ))
                for i in range(self.particles):
                    kalman_gain[i] = coin_state["Kalman_gains"][max_inds[i], inds[i]]
                coin_state["Kalman_gain_given_cstar2"] = np.mean(kalman_gain)
                    
        if "state_given_cstar2" in self.store:
            if coin_state["trial"] > 0:
                max_inds = np.argmax(coin_state["predicted_probabilities"], axis=0)
                inds = np.arange(self.particles)
                
                state = np.zeros((self.particles, ))
                
                for i in range(len(max_inds)):
                    state[i] = coin_state["state_mean"][max_inds[i], inds[i]]
                coin_state["state_given_cstar2"] = np.mean(state)
        
        if "predicted_probability_cstar3" in self.store:
            coin_state["predicted_probability_cstar3"] = np.mean(np.max(coin_state["predicted_probabilities"], axis=0))
        
        return coin_state
    
    def predict_states(self, coin_state: Dict[str, Any]):
        # propagate states (Equation 3)
        coin_state["state_mean"] = coin_state["retention"] * coin_state["state_filtered_mean"] + coin_state["drift"]
        # TODO: verify if we need to add the variance associated with the drift
        coin_state["state_var"] = np.square(coin_state["retention"]) * coin_state["state_filtered_var"] + np.square(self.sigma_process_noise) # + 1 / self.prior_precision_drift
        
        # index of novel states
        inds_1 = coin_state["C"] # TODO: check if this is correct now that we are initialising contexts from 1
        inds_2 = np.arange(self.particles)
        
        # novel states are initialised to follow the stationary distribution (under Gaussian LDS)
        for i in range(self.particles):
            coin_state["state_mean"][inds_1[i], inds_2[i]] = coin_state["drift"][inds_1[i], inds_2[i]] / (1 - coin_state["retention"][inds_1[i], inds_2[i]])
            coin_state["state_var"][inds_1[i], inds_2[i]] = np.square(self.sigma_process_noise) / (1 - np.square(coin_state["retention"][inds_1[i], inds_2[i]]))
        
        # predict state (marginalise over contexts and particles)
        # mean of the distribution (sum over all contexts and all particles)
        coin_state["average_state"] = np.sum(coin_state["predicted_probabilities"] * coin_state["state_mean"]) / self.particles
        
        if "explicit" in self.store:
            # given posterior distribution.
            if coin_state["trial"] == 0:
                coin_state["explicit"] = np.mean(coin_state["state_mean"][0, :])
            else:
                max_inds = np.argmax(coin_state["responsibilities"], axis=0)
                inds = np.arange(self.particles)
                state_mean = np.zeros((self.particles, ))
                for i in range(self.particles):
                    state_mean[i] = coin_state["state_mean"][max_inds[i], inds[i]]
                coin_state["explicit"] = np.mean(state_mean)
        
        if "state_given_cstar3" in self.store:
            # given posterior predictive distribution.
            max_inds = np.argmax(coin_state["predicted_probabilities"], axis=0)
            inds = np.arange(self.particles)
            state_mean = np.zeros((self.particles, ))
            for i in range(self.particles):
                state_mean[i] = coin_state["state_mean"][max_inds[i], inds[i]]
            coin_state["state_given_cstar3"] = np.mean(state_mean)
        
        return coin_state
    
    def predict_state_feedback(self, coin_state: Dict[str, Any]):
        # predict state feedback for each context (potential non-trivial context-specific bias term in visuo-motor tasks) (eq. 19)
        if "bias" in coin_state:
            coin_state["state_feedback_mean"] = coin_state["state_mean"] + coin_state["bias"]
        else:
            coin_state["state_feedback_mean"] = coin_state["state_mean"]
        
        # variance of state feedback prediction for each context
        if "bias" in coin_state:
            coin_state["state_feedback_var"] = coin_state["state_var"] + np.square(coin_state["sigma_observation_noise"]) + 1 / self.prior_precision_bias # (eq. 19)
        else:
            coin_state["state_feedback_var"] = coin_state["state_var"] + np.square(coin_state["sigma_observation_noise"]) # (eq. 19)
        
        coin_state = self.compute_marginal_distribution(coin_state)
        
        # predict marginalised state feedback (marginalise over contexts and particles)
        # mean of the distribution
        coin_state["motor_output"] = np.sum(coin_state["predicted_probabilities"] * coin_state["state_feedback_mean"]) / self.particles
        
        if "implicit" in self.store:
            coin_state["implicit"] = coin_state["motor_output"] - coin_state["average_state"]
        
        # sensnory and motor noise
        coin_state["sensory_noise"] = self.sigma_sensory_noise * np.random.randn()
        coin_state["motor_noise"] = self.sigma_motor_noise * np.random.randn()
        
        # state feedback
        coin_state["state_feedback"] = self.perturbations[coin_state["trial"]] + coin_state["sensory_noise"] + coin_state["motor_noise"]
        
        # state feedback prediction error
        coin_state["prediction_error"] = coin_state["state_feedback"] - coin_state["state_feedback_mean"]
        
        return coin_state
    
    def resample_particles(self, coin_state: Dict[str, Any]):
        # p(y_t|c_t)
        coin_state["probability_state_feedback"] = norm(coin_state["state_feedback_mean"], np.sqrt(coin_state["state_feedback_var"])).pdf(coin_state["state_feedback"])
        
        if coin_state["feedback_observed"][coin_state["trial"]]:
            if coin_state["cues_exist"]:
                # log p(y_t, q_t, c_t)
                p_c = np.log(coin_state["prior_probabilities"]) + np.log(coin_state["cue_probabilities"]) + np.log(coin_state["probability_state_feedback"])
            else:
                # log p(y_t, c_t)
                p_c = np.log(coin_state["prior_probabilities"]) + np.log(coin_state["probability_state_feedback"])
        else:
            if coin_state["cues_exist"]:
                # log p(q_t, c_t)
                p_c = np.log(coin_state["prior_probabilities"]) + np.log(coin_state["probability_cue"])
            else:
                # log p(c_t)
                p_c = np.log(coin_state["prior_probabilities"])
        
        log_weights = log_sum_exp(p_c) # log p(y_t, q_t) (marginalise out contexts)

        p_c = p_c - log_weights # log p(c_t|y_t, q_t)
        
        # weights for resampling
        w = np.exp(log_weights - log_sum_exp(log_weights.T, axis=0))
        
        # draw indices of particles to propagate
        if coin_state["feedback_observed"][coin_state["trial"]] or coin_state["cues_exist"]:
            coin_state["inds_resampled"] = systematic_resampling(w)
        else:
            coin_state["inds_resampled"] = np.arange(self.particles)
        
        # store variables of the predictive distributions (optional)
        # these variables are stored before resampling (so that they do not depend on the current state feedback)
        variables_stored_before_resampling = [
            "predicted_probabilities", 
            "state_feedback_mean", 
            "state_feedback_var", 
            "state_mean", 
            "state_var", 
            "Kalman_gain_given_cstar2", 
            "state_given_cstar2", 
        ]
        
        for i in range(len(self.store)):
            variable = self.store[i]
            if (variable in variables_stored_before_resampling) and (variable in coin_state):
                coin_state = self.store_function(coin_state, variable)
        
        # resample particles
        inds_resampled = coin_state["inds_resampled"]
        
        coin_state["previous_context"] = coin_state["context"][inds_resampled]
        coin_state["prior_probabilities"] = coin_state["prior_probabilities"][:, inds_resampled]
        coin_state["predicted_probabilities"] = coin_state["predicted_probabilities"][:, inds_resampled]
        coin_state["responsibilities"] = np.exp(p_c[:, inds_resampled])
        coin_state["C"] = coin_state["C"][inds_resampled]
        coin_state["state_mean"] = coin_state["state_mean"][:, inds_resampled]
        coin_state["state_var"] = coin_state["state_var"][:, inds_resampled]
        coin_state["prediction_error"] = coin_state["prediction_error"][:, inds_resampled]
        coin_state["state_feedback_var"] = coin_state["state_feedback_var"][:, inds_resampled]
        coin_state["probability_state_feedback"] = coin_state["probability_state_feedback"][:, inds_resampled]
        coin_state["global_transition_probabilities"] = coin_state["global_transition_probabilities"][:, inds_resampled]
        coin_state["n_context"] = coin_state["n_context"][:, :, inds_resampled]
        coin_state["previous_state_filtered_mean"] = coin_state["state_filtered_mean"][:, inds_resampled]
        coin_state["previous_state_filtered_var"] = coin_state["state_filtered_var"][:, inds_resampled]
        
        if coin_state["cues_exist"]:
            coin_state["global_cue_probabilities"] = coin_state["global_cue_probabilities"][:, inds_resampled]
            coin_state["n_cue"] = coin_state["n_cue"][:, :, inds_resampled]
        
        coin_state["retention"] = coin_state["retention"][:, inds_resampled]
        coin_state["drift"] = coin_state["drift"][:, inds_resampled]
        coin_state["dynamics_ss_1"] = coin_state["dynamics_ss_1"][:, inds_resampled, :]
        coin_state["dynamics_ss_2"] = coin_state["dynamics_ss_2"][:, inds_resampled, :, :]
        
        if self.infer_bias:
            coin_state["bias"] = coin_state["bias"][:, inds_resampled]
            coin_state["bias_ss_1"] = coin_state["bias_ss_1"][:, inds_resampled]
            coin_state["bias_ss_2"] = coin_state["bias_ss_2"][:, inds_resampled]
        
        return coin_state
    
    def sample_context(self, coin_state: Dict[str, Any]):
        # sample context
        coin_state["context"] = np.sum(np.random.rand(self.particles) > np.cumsum(coin_state["responsibilities"], axis=0), axis=0) + 1
        
        # increment the context count
        coin_state["p_new_x"] = np.where(coin_state["context"] > coin_state["C"])[0]
        coin_state["p_old_x"] = np.where(coin_state["context"] <= coin_state["C"])[0]
        coin_state["C"][coin_state["p_new_x"]] = coin_state["C"][coin_state["p_new_x"]] + 1 # increment
        
        p_beta_x = coin_state["p_new_x"][coin_state["C"][coin_state["p_new_x"]] != self.max_contexts]
        inds = coin_state["context"][p_beta_x] - 1
        
        # sample the next stick-breaking weight
        sb_weight = np.random.beta(1, self.gamma_context * np.ones((len(p_beta_x), )))
        
        # update the global transition distribution
        coin_state["global_transition_probabilities"][inds+1, p_beta_x] = coin_state["global_transition_probabilities"][inds, p_beta_x] * (1 - sb_weight)
        coin_state["global_transition_probabilities"][inds, p_beta_x] = coin_state["global_transition_probabilities"][inds, p_beta_x] * sb_weight
        
        if coin_state["cues_exist"]:
            if self.cues[coin_state["trial"]-1] > coin_state["Q"]:
                # increment the cue context count
                coin_state["Q"] += 1
                
                # sample the next stick-breaking weight
                sb_weight = np.random.beta(1, self.gamma_cue * np.ones((self.particles, )))
                
                coin_state["global_cue_probabilities"][coin_state["Q"], :] = coin_state["global_cue_probabilities"][coin_state["Q"]-1, :] * (1 - sb_weight)
                coin_state["global_cue_probabilities"][coin_state["Q"]-1, :] = coin_state["global_cue_probabilities"][coin_state["Q"]-1, :] * sb_weight
                
        return coin_state
    
    def update_belief_about_states(self, coin_state: Dict[str, Any]):
        # algorithm 3 (Kalman filtering)
        coin_state["Kalman_gains"] = coin_state["state_var"] / coin_state["state_feedback_var"]
        if coin_state["feedback_observed"][coin_state["trial"]]:
            coin_state["state_filtered_mean"] = coin_state["state_mean"] + coin_state["Kalman_gains"] * coin_state["prediction_error"] * coin_state["H"][coin_state["context"]-1, :].T
            coin_state["state_filtered_var"] = (1 - coin_state["Kalman_gains"] * coin_state["H"][coin_state["context"]-1, :].T) * coin_state["state_var"]
        else:
            coin_state["state_filtered_mean"] = coin_state["state_mean"]
            coin_state["state_filtered_var"] = coin_state["state_var"]
        
        return coin_state
    
    def sample_states(self, coin_state: Dict[str, Any]):
        n_new_x = len(coin_state["p_new_x"])
        inds_old_x = [coin_state["context"][coin_state["p_old_x"]]-1, coin_state["p_old_x"]]
        inds_new_x = [coin_state["context"][coin_state["p_new_x"]]-1, coin_state["p_new_x"]]
        
        # for states that have been observed before, sample x_{t-1}, then sample x_t given x_{t-1}
        # sample x_{t-1} using a fixed-lag (lag 1) forward-backward smoother
        g = coin_state["retention"] * coin_state["previous_state_filtered_var"] / coin_state["state_var"]
        m = coin_state["previous_state_filtered_mean"] + g * (coin_state["state_filtered_mean"] - coin_state["state_mean"])
        v = coin_state["previous_state_filtered_var"] + g * (coin_state["state_filtered_var"] - coin_state["state_var"]) * g
        # sample from the smoothing posterior distribution
        coin_state["previous_x_dynamics"] = m + np.emath.sqrt(v) * np.random.randn(self.max_contexts+1, self.particles)
        
        # sample x_t conitioned on x_{t-1} and y_t
        if coin_state["feedback_observed"][coin_state["trial"]]:
            w = (coin_state["retention"] * coin_state["previous_x_dynamics"] + coin_state["drift"]) / np.square(self.sigma_process_noise) + \
                coin_state["H"][coin_state["context"]-1, :].T / np.square(coin_state["sigma_observation_noise"]) * (coin_state["state_feedback"] - coin_state["bias"])
            v = 1 / (1 / np.square(self.sigma_process_noise) + coin_state["H"][coin_state["context"]-1, :].T / np.square(coin_state["sigma_observation_noise"])) 
            # TODO: verify if the -1 is correct!
        else:
            w = (coin_state["retention"] * coin_state["previous_x_dynamics"] + coin_state["drift"]) / np.square(self.sigma_process_noise)
            v = 1 / (1 / np.square(self.sigma_process_noise))
        coin_state["x_dynamics"] = v * w + np.sqrt(v) * np.random.randn(self.max_contexts+1, self.particles)

        x_sample_novel = coin_state["state_filtered_mean"][inds_new_x[0], inds_new_x[1]] + np.sqrt(coin_state["state_filtered_var"][inds_new_x[0], inds_new_x[1]]) * \
            np.random.randn(n_new_x)
        
        coin_state["x_bias"] = np.concatenate([coin_state["x_dynamics"][inds_old_x[0], inds_old_x[1]], x_sample_novel], axis=-1)
        coin_state["inds_observed"] = [
            np.concatenate([inds_old_x[0], inds_new_x[0]]), 
            np.concatenate([inds_old_x[1], inds_new_x[1]]), 
        ]
        
        return coin_state
    
    def update_sufficient_statistics_for_parameters(self, coin_state: Dict[str, Any]):
        # update sufficient statistics for the parameters of the global transition probabilities
        coin_state = self.update_sufficient_statistics_global_transition_probabilities(coin_state)
        
        # update sufficient statistics for the parameters of the global cue probabilities
        if coin_state["cues_exist"]:
            coin_state = self.update_sufficient_statistics_global_cue_probabilities(coin_state)
            
        if coin_state["trial"] > 1:
            # update sufficient for the parameters of the state dynamics function
            coin_state = self.update_sufficient_statistics_dynamics(coin_state)
        
        if self.infer_bias and (coin_state["feedback_observed"][coin_state["trial"]]):
            coin_state = self.update_sufficient_statistics_bias(coin_state)
        
        return coin_state
    
    def store_variables(self, coin_state: Dict[str, Any]):
        if "Kalman_gain_given_cstar1" in self.store:
            max_inds = np.argmax(coin_state["responsibilities"], axis=0)
            coin_state["Kalman_gain_given_cstar1"] = np.mean(coin_state["Kalman_gains"][max_inds, np.arange(self.particles)], axis=0)
        
        if "predicted_probability_cstar1" in self.store:
            max_inds = np.argmax(coin_state["responsibilities"], axis=0)
            coin_state["predicted_probability_cstar1"] = np.mean(coin_state["predicted_probabilities"][max_inds, np.arange(self.particles)], axis=0)
        
        if "state_given_cstar1" in self.store:
            max_inds = np.max(coin_state["responsibilities"], axis=0)
            coin_state["state_given_cstar"] = np.mean(coin_state["state_mean"][max_inds, np.arange(self.particles)], axis=0)
        
        variables_stored_before_resampling = [
            "predicted_probabilities", 
            "state_feedback_mean", 
            "state_feedback_var", 
            "state_mean", 
            "state_var", 
            "Kalman_gain_given_cstar2", 
            "state_given_cstar2", 
        ]
        for i in range(len(self.store)):
            variable = self.store[i]
            if variable not in variables_stored_before_resampling:
                coin_state = self.store_function(coin_state, variable)
        
        return coin_state
        
    def sample_parameters(self, coin_state: Dict[str, Any]):
        # sample global transition probabilities
        coin_state = self.sample_global_transition_probabilities(coin_state)
        
        # update local context transition probability matrix
        coin_state = self.update_local_transition_matrix(coin_state)
        
        if coin_state["cues_exist"]:
            # sample global cue probabilities
            coin_state = self.sample_global_cue_probabilities(coin_state)
            
            # update local cue probability matrix
            coin_state = self.update_local_cue_matrix(coin_state)
        
        # sample parameters of the state dynamics function
        coin_state = self.sample_dynamics(coin_state)
        
        # sample parameters of the observation function
        if self.infer_bias:
            coin_state = self.sample_bias(coin_state)
        else:
            coin_state["bias"] = 0

        return coin_state
    
    def sample_global_transition_probabilities(self, coin_state: Dict[str, Any]):
        # global context DP (eq. 7)
        if coin_state["trial"] == 0:
            coin_state["global_transition_probabilities"] = np.zeros((self.max_contexts+1, self.particles))
            coin_state["global_transition_probabilities"][0, :] = 1.
        else:
            # sample the number of tables in restaurant i serving dish j
            if not self.sample_crf_stirling:
                coin_state["m_context"] = randnumtable_array(
                    self.alpha_context * coin_state["global_transition_probabilities"][None] + coin_state["kappa"] * np.eye(self.max_contexts+1)[..., None], 
                    coin_state["n_context"], # (max_contexts+1, max_contexts+1, particles)
                )
            else:
                coin_state["m_context"], _, _, _ = sample_num_tables_CRF(
                    self.alpha_context * coin_state["global_transition_probabilities"][None] + coin_state["kappa"] * np.eye(self.max_contexts+1)[..., None], 
                    coin_state["n_context"], # (max_contexts+1, max_contexts+1, particles)
                )
            
            
            # sample the number of tables in restaurant i considering dish j
            m_context_bar = coin_state["m_context"].astype(float)
            
            if self.rho_context > 0:
                inds_1 = np.tile(np.arange(self.max_contexts+1)[None], (self.particles, 1)).T
                inds_2 = np.tile(np.arange(self.max_contexts+1)[None], (self.particles, 1)).T
                inds_3 = np.tile(np.arange(self.particles)[None], (self.max_contexts+1, 1))
                
                m_context = np.zeros((self.max_contexts+1, self.particles))
                for i in range(self.max_contexts+1):
                    for j in range(self.particles):
                        m_context[i, j] = coin_state["m_context"][inds_1[i, j], inds_2[i, j], inds_3[i, j]]
                
                non_zero_inds_1, non_zero_inds_2 = np.where(m_context != 0)
                
                p = self.rho_context / (self.rho_context + coin_state["global_transition_probabilities"][non_zero_inds_1, non_zero_inds_2] \
                    * (1 - self.rho_context))
                
                inds_1 = inds_1[non_zero_inds_1, non_zero_inds_2]
                inds_2 = inds_2[non_zero_inds_1, non_zero_inds_2]
                inds_3 = inds_3[non_zero_inds_1, non_zero_inds_2]
                
                # m_context_bar[inds_1, inds_2, inds_3] = coin_state["m_context"][inds_1, inds_2, inds_3] - \
                #     random_binomial(p, coin_state["m_context"][inds_1, inds_2, inds_3])
                m_context_bar[inds_1, inds_2, inds_3] = coin_state["m_context"][inds_1, inds_2, inds_3] - \
                    np.random.binomial(coin_state["m_context"][inds_1, inds_2, inds_3], p)
            
            # handling boundary condition
            m_context_bar[0, 0, m_context_bar[0, 0, :] == 0] = 1
            
            # sample beta
            inds = np.where(coin_state["C"] != self.max_contexts)[0]
            inds_C = coin_state["C"][inds] # + 1
            global_transition_posterior_dirichlet_param = np.sum(m_context_bar, axis=0)
            for i in range(len(inds)):
                global_transition_posterior_dirichlet_param[inds_C[i], inds[i]] = self.gamma_context
            
            coin_state["global_transition_probabilities"] = random_dirichlet(global_transition_posterior_dirichlet_param)

        return coin_state
    
    def sample_global_cue_probabilities(self, coin_state: Dict[str, Any]):
        # global cue DP (eq. 9)
        # similar to sample_global_transition_probabilities but without self-transition bias ("loyalty customers")
        if coin_state["trial"] == 0:
            coin_state["global_cue_probabilities"] = np.zeros((np.max(self.cues)+1, self.particles))
            coin_state["global_cue_probabilities"][0, :] = 1.0
        else:
            # sample the number of tables in restaurant i serving dish j
            if not self.sample_crf_stirling:
                coin_state["m_cue"] = randnumtable_array(
                    np.tile(self.alpha_cue * coin_state["global_cue_probabilities"][None], (self.max_contexts+1, 1, 1)), 
                    coin_state["n_cue"], 
                )
            else:
                coin_state["m_cue"] = sample_num_tables_CRF(
                    np.tile(self.alpha_cue * coin_state["global_cue_probabilities"][None], (self.max_contexts+1, 1, 1)), 
                    coin_state["n_cue"], 
                )
            
            # sample beta_e
            coin_state["global_cue_posterior"] = np.reshape(np.sum(coin_state["m_cue"], axis=0), (self.cues+1, self.particles))
            coin_state["global_cue_posterior"][coin_state["Q"]+1, :] = self.gamma_cue
            
            coin_state["global_cue_probabilities"] = random_dirichlet(coin_state["global_cue_posterior"])
        
        return coin_state
    
    def update_local_transition_matrix(self, coin_state: Dict[str, Any]):
        # local context DP (eq. 8)
        coin_state["local_transition_matrix"] = np.reshape(
            self.alpha_context * coin_state["global_transition_probabilities"], 
            [1, self.max_contexts + 1, self.particles], 
        ) + coin_state["n_context"] + coin_state["kappa"] * np.eye(self.max_contexts+1)[..., None] # stickiness
        coin_state["local_transition_matrix"] = coin_state["local_transition_matrix"] / np.sum(
            coin_state["local_transition_matrix"], axis=1, keepdims=True, 
        )
        
        # remove contexts with zero mass under the global transition distribution
        zero_mass_contexts = (np.reshape(coin_state["global_transition_probabilities"], (self.max_contexts+1, 1, self.particles)) > 0)
        coin_state["local_transition_matrix"] = coin_state["local_transition_matrix"] * zero_mass_contexts
        
        # compute stationary context probabilities if necessary
        if ("stationary_probabilities" in self.store) and (coin_state["trial"] > 0):
            coin_state["stationary_probabilities"] = np.zeros((self.max_contexts+1, self.particles))
            
            for p in range(self.particles):
                C = coin_state["C"][p]
                transmat = coin_state["local_transition_matrix"][:(C+1), :(C+1), p]
                coin_state["stationary_probabilities"][:(C+1), p] = stationary_distribution(transmat)
        
        return coin_state
    
    def update_local_cue_matrix(self, coin_state: Dict[str, Any]):
        # local cue DP (eq. 9)
        coin_state["local_cue_matrix"] = np.reshape(
            self.alpha_cue * coin_state["global_cue_probabilities"], 
            (1, np.max(self.cues) + 1, self.particles)
        ) + coin_state["n_cue"]
        coin_state["local_cue_matrix"] = coin_state["local_cue_matrix"] / np.sum(coin_state["local_cue_matrix"], axis=0, keepdims=True)
        
        zero_mass_contexts = (np.reshape(coin_state["global_transition_probabilities"], (self.max_contexts+1, 1, self.particles)) > 0)
        coin_state["local_cue_matrix"] = coin_state["local_cue_matrix"] * zero_mass_contexts
        
        return coin_state
    
    def sample_dynamics(self, coin_state: Dict[str, Any]):
        # sample state retention and drift parameters (eqs. S21 and S22)
        
        # prior mean and precision matrix (eq. 10)
        dynamics_mean = np.array([self.prior_mean_retention, 0])
        dynamics_lambda = np.eye(2) * np.array([self.prior_precision_retention, self.prior_precision_drift])
        
        # update the parameters of the posterior
        coin_state["dynamics_covariance"] = per_slice_invert(
            dynamics_lambda[..., None] + 
            np.transpose(np.reshape(coin_state["dynamics_ss_2"], ((self.max_contexts+1)*self.particles, 2, 2)), (1, 2, 0)) / (self.sigma_process_noise ** 2)
        )
        coin_state["dynamics_mean"] = per_slice_multiply(
            coin_state["dynamics_covariance"], 
            (np.dot(dynamics_lambda, dynamics_mean)[:, None] + 
            np.reshape(coin_state["dynamics_ss_1"], ((self.max_contexts+1) * self.particles, 2)).T / 
            (self.sigma_process_noise ** 2))[..., None]
        )
        
        # sample the parameters of the state dynamics function
        dynamics = random_truncated_bivariate_normal(
            coin_state["dynamics_mean"], 
            coin_state["dynamics_covariance"], 
        )
        coin_state["retention"] = np.reshape(dynamics[0, :], (self.max_contexts+1, self.particles))
        coin_state["drift"] = np.reshape(dynamics[1, :], (self.max_contexts+1, self.particles))
        
        coin_state["dynamics_mean"] = np.reshape(coin_state["dynamics_mean"], [2, self.max_contexts+1, self.particles])
        coin_state["dynamics_covariance"] = np.reshape(
            coin_state["dynamics_covariance"], (2, 2, self.max_contexts+1, self.particles)
        )
        
        return coin_state
    
    def sample_bias(self, coin_state: Dict[str, Any]):
        bias_mean = 0.0
        
        coin_state["bias_var"] = 1. / (self.prior_precision_bias + coin_state["bias_ss_2"] / (coin_state["sigma_observation_noise"] ** 2))
        coin_state["bias_mean"] = coin_state["bias_var"] * (
            self.prior_precision_bias * bias_mean + coin_state["bias_ss_1"] / (coin_state["sigma_observation_noise"] ** 2)
        )
        
        coin_state["bias"] = random_univariate_normal(coin_state["bias_mean"], coin_state["bias_var"])
        
        return coin_state
    
    def compute_marginal_distribution(self, coin_state: Dict[str, Any]):
        if "state_distribution" in self.store:
            # predict state (marginalisation over contexts and particles)
            x = np.reshape(self.state_values, (1, 1, self.state_values.size))
            mu = coin_state["state_mean"]
            std = np.sqrt(coin_state["state_var"])
            coin_state["state_distribution"] = np.sum(coin_state["predicted_probabilities"][..., None] * norm(mu[..., None], std[..., None]).pdf(x), axis=(0, 1)) / self.particles
        
        if "bias_distribution" in self.store:
            x = np.reshape(self.bias_values, (1, 1, self.bias_values.size))
            mu = coin_state["bias_mean"]
            std = np.sqrt(coin_state["bias_var"])
            coin_state["bias_distribution"] = np.sum(coin_state["predicted_probabilities"][..., None] * norm(mu[..., None], std[..., None]).pdf(x), axis=(0, 1)) / self.particles
            
        if "state_feedback_distribution" in self.store:
            x = np.reshape(self.state_feedback_values, (1, 1, self.state_feedback_values.size))
            mu = coin_state["state_feedback_mean"]
            std = np.sqrt(coin_state["state_feedback_var"])
            coin_state["state_feedback_distribution"] = np.sum(coin_state["predicted_probabilities"][..., None] * norm(mu[..., None], std[..., None]).pdf(x), axis=(0, 1)) / self.particles
            
        return coin_state
    
    def store_function(self, coin_state: Dict[str, Any], variable: str):
        if variable in ["Kalman_gain_given_cstar2", "state_given_cstar2"]:
            store_on = "previous_trial"
        else:
            store_on = "current_trial"
            
        if "stored" not in coin_state:
            coin_state["stored"] = {}
        
        if ((coin_state["trial"] == 0) and (store_on == "current_trial")) or ((coin_state["trial"] == 1) and (store_on == "previous_trial")):
            s = list(coin_state[variable].shape) + [coin_state["num_trials"]]
            coin_state["stored"][variable] = np.ones(s) * np.nan
            
        if store_on == "current_trial":
            trial = coin_state["trial"]
        elif store_on == "previous_trial":
            trial = coin_state["trial"] - 1
        
        coin_state["stored"][variable][..., trial] = coin_state[variable]
        
        return coin_state

    def update_sufficient_statistics_global_transition_probabilities(self, coin_state: Dict[str, Any]):
        inds_1 = coin_state["previous_context"] - 1
        inds_2 = coin_state["context"] -1 # TODO: is the -1 right?
        inds_3 = np.arange(self.particles)
        
        for i in range(self.particles):
            coin_state["n_context"][inds_1[i], inds_2[i], inds_3[i]] = coin_state["n_context"][inds_1[i], inds_2[i], inds_3[i]] + 1

        return coin_state
    
    def update_sufficient_statistics_global_cue_probabilities(self, coin_state: Dict[str, Any]):
        inds_1 = coin_state["context"] - 1 # TODO: is the -1 right?
        inds_2 = self.cues[coin_state["trial"]] * np.ones((self.particles, ))
        inds_3 = np.arange(self.particles)
        
        for i in range(self.particles):
            coin_state["n_cue"][inds_1[i], inds_2[i], inds_3[i]] = coin_state["n_cue"][inds_1[i], inds_2[i], inds_3[i]] + 1

        return coin_state
    
    def update_sufficient_statistics_dynamics(self, coin_state: Dict[str, Any]):
        # eq. (S20)
        # augment the state vector: x_{t-1} -> [x_{t-1}, 1]
        x_a = np.ones((self.max_contexts+1, self.particles, 2))
        x_a[:, :, 0] = coin_state["previous_x_dynamics"]
        
        # identify states that are not novel
        I = np.reshape(np.sum(coin_state["n_context"], axis=1), (self.max_contexts+1, self.particles)) > 0
        
        SS = coin_state["x_dynamics"][..., None] * x_a # x_t * [x_{t-1}, 1]
        coin_state["dynamics_ss_1"] = coin_state["dynamics_ss_1"] + SS * I[..., None]
        
        SS = x_a[..., None] * x_a[:, :, None, :]
        coin_state["dynamics_ss_2"] = coin_state['dynamics_ss_2'] + SS * I[..., None, None]
        
        # Make sure the variables are not complex
        assert ~(np.imag(coin_state["dynamics_ss_1"]) != 0).any()
        assert ~(np.imag(coin_state["dynamics_ss_2"]) != 0).any()
        if coin_state["dynamics_ss_1"].dtype == np.complex128 and ~(np.imag(coin_state["dynamics_ss_1"]) != 0).any():
            coin_state["dynamics_ss_1"] = np.float64(coin_state["dynamics_ss_1"])
        if coin_state["dynamics_ss_2"].dtype == np.complex128 and ~(np.imag(coin_state["dynamics_ss_2"]) != 0).any():
            coin_state["dynamics_ss_2"] = np.float64(coin_state["dynamics_ss_2"])
        return coin_state
    
    def update_sufficient_statistics_bias(self, coin_state: Dict[str, Any]):
        coin_state["bias_ss_1"][coin_state["inds_observed"][0], coin_state["inds_observed"][1]] += coin_state["state_feedback"] - coin_state["x_bias"]
        coin_state["bias_ss_2"][coin_state["inds_observed"][0], coin_state["inds_observed"][1]] += 1
        
        return coin_state
    
    def set_store_property_for_plots(self):
        temp = []
        if self.plot_state_given_context:
            temp.extend(["state_mean", "state_var"])
        if self.plot_predicted_probabilities:
            temp.append("predicted_probabilities")
        if self.plot_responsibilities:
            temp.append("responsibilities")
        if self.plot_stationary_probabilities:
            temp.append("stationary_probabilities")
        if self.plot_retention_given_context:
            temp.extend(["dynamics_mean", "dynamics_covar"])
        if self.plot_drift_given_context:
            temp.extend(["dynamics_mean", "dynamics_covar"])
        if self.plot_bias_given_context:
            if self.infer_bias:
                temp.extend(["bias_mean", "bias_var"])
            else:
                raise ValueError
        if self.plot_global_transition_probabilities:
            temp.append("global_transition_probabilities")
        if self.plot_local_transition_probabilities:
            temp.append("local_transition_probabilities")
        if self.plot_local_cue_probabilities:
            if self.cues is None or len(self.cues) == 0:
                raise ValueError
            else:
                temp.append("local_cue_matrix")
        if self.plot_global_cue_probabilities:
            if self.cues is None or len(self.cues) == 0:
                raise ValueError
            else:
                temp.append("global_cue_posterior")
        if self.plot_state:
            temp.extend(["state_distribution", "average_state"])
        if self.plot_average_state:
            temp.append("average_state")
        if self.plot_bias:
            if self.infer_bias:
                temp.extend(["bias_distribution", "implicit"])
            else:
                raise ValueError
        if self.plot_average_bias:
            temp.append("implicit")
        if self.plot_state_feedback:
            temp.append("state_feedback_distribution")
        if self.plot_explicit_component:
            temp.append("explicit")
        if self.plot_implicit_component:
            temp.append("implicit")
        if self.plot_Kalman_gain_given_cstar1:
            temp.append("Kalman_gain_given_cstar1")
        if self.plot_predicted_probability_cstar1:
            temp.append("predicted_probability_cstar1")
        if self.plot_state_given_cstar1:
            temp.append("state_given_cstar1")
        if self.plot_Kalman_gain_given_cstar2:
            temp.append("Kalman_gain_given_cstar2")
        if self.plot_state_given_cstar2:
            temp.append("state_given_cstar2")
        if self.plot_predicted_probability_cstar3:
            temp.append("predicted_probability_cstar3")
        if self.plot_state_given_cstar3:
            temp.append("state_given_cstar3")
        
        if len(temp) > 0:
            temp.extend(["context", "inds_resampled"])
        
        for i in range(len(temp)):
            if temp[i] not in self.store:
                self.store.append(temp[i])

    def plot_coin(self, S: Dict[str, Any]):
        variables_requiring_context_relabelling = [
            "state_given_context", 
            "predicted_probabilities", 
            "responsibilties", 
            "stationary_probabilities", 
            "retention_given_context", 
            "drift_given_context", 
            "bias_given_context", 
            "global_transition_probabilities", 
            "local_transition_probabilities", 
            "global_cue_probabilities", 
            "local_cue_probabilities", 
        ]
        
        for i in range(len(variables_requiring_context_relabelling)):
            s = variables_requiring_context_relabelling[i]
            if self.__dict__[f"plot_{s}"]:
                P, S, optim_assignment, from_unique, c_seq, C = self.find_optimal_context_labels(S)
                P, _ = self.compute_variables_for_plotting(P, S, optim_assignment, from_unique, c_seq, C)
                break
            elif i == (len(variables_requiring_context_relabelling)-1):
                P = self.preallocate_memory([]) # TODO: how does it handle empty dictionary?
                P = self.integrate_over_runs(P, S)
        
        self.generate_figures(P)

    def find_optimal_context_labels(self, S: Dict[str, Any]):
        inds_resampled = self.resample_inds(S)
        context_sequence = self.context_sequence(S, inds_resampled)
        C, _, _, mode_number_of_contexts = self.posterior_number_of_contexts(context_sequence, S)
        
        P = {}
        P["mode_number_of_contexts"] = mode_number_of_contexts
        
        # context label permutations
        # flipping 
        L = np.array(list(permutations(np.arange(1, np.max(mode_number_of_contexts)+1)))[::-1])
        L = np.transpose(L[None], (2, 0, 1)) # (C, 1, C!)
        n_perms = factorial(np.max(mode_number_of_contexts)) # + 1) # TODO: do we need +1?
        
        num_trials = len(self.perturbations)
        
        f = {}
        to_unique = {}
        from_unique = {}
        optimal_assignment = {}
        
        with trange(num_trials, desc="Finding optimal context labels") as pbar:
            for i in pbar:
        # for i in range(num_trials):
        #     if np.mod(i+1, 50) == 0:
        #         print(f"Finding optimal context labels (trial = {i+1})")
            
                # exclude sequences for which C > max(mode_number_of_context) as these sequences
                # (and their descendents) will never be analysed
                f[i] = np.where(C[:, i] <= np.max(P["mode_number_of_contexts"]))[0]
                
                # identify unique sequences (to avoid performing the same computations multiple times)
                unique_seqs, inds, reverse_inds = np.unique(
                    context_sequence[i][f[i]], axis=0, return_index=True, return_inverse=True
                )
                to_unique[i] = inds
                from_unique[i] = reverse_inds
                
                n_sequences = len(unique_seqs)
                
                # identify particles that have the same number of contexts as the most common number of
                # contexts (only these particles will be analysed)
                valid_particle_inds = (C[f[i], i] == P["mode_number_of_contexts"][i])
                
                if i == 0:
                    # hamming distances on trial 0
                    # dimension 2 of H considers all possible label permutations
                    H = (L[[0], :, :] != 1) * 1.0 # (1, 1, num_permutations)
                else:
                    # identify a valid parent of each unique sequence
                    # i.e., a sequence on the previous trial that is identical up to the previous trial
                    inds, _ = np.where(f[i-1][:, None] == inds_resampled[f[i][to_unique[i]], i][None])
                    parent = from_unique[i-1][inds]
                    
                    # pass Hamming distances from parents to children
                    inds_1 = np.tile(parent[:, None, None], [1, n_sequences, n_perms])
                    inds_2 = np.tile(parent[None, :, None], [n_sequences, 1, n_perms])
                    inds_3 = np.tile(np.arange(n_perms)[None, None], [n_sequences, n_sequences, 1])
                    
                    H_new = np.zeros((n_sequences, n_sequences, n_perms))
                    
                    for ii in range(n_sequences):
                        for jj in range(n_sequences):
                            for kk in range(n_perms):
                                H_new[ii, jj, kk] = H[inds_1[ii, jj, kk], inds_2[ii, jj, kk], inds_3[ii, jj, kk]]
                    
                    H = H_new.copy()
                    
                    # recursively update Hamming distances
                    # dimension 2 of H considers all possible label permutations
                    for seq in range(n_sequences):
                        H[seq:, [seq], :] = H[seq:, [seq], :] + (unique_seqs[seq, -1] != L[unique_seqs[seq:, -1]-1, :, :]) * 1.0
                        H[seq, seq:, :] = H[seq:, seq, :] # by symmetry of Hamming distance\
            
                # compute the Hamming distance between each pair of sequences (after optimally permuting labels)
                H_optimal = np.min(H, axis=2)
                
                # count the number of times each unique sequence occurs
                sequence_count = np.sum(
                    from_unique[i][valid_particle_inds][:, None] == np.arange(len(unique_seqs))[None], 
                    axis=0, 
                )
                
                # compute the mean optimal Hamming distance of each sequence to all other sequences.
                # the distance from sequence i to sequence j is weighted by the number of times sequence j occurs.
                # if i == j, this weight is reduced by 1 so that the distance from one instance of sequence i to itself is ignore.
                H_mean = np.mean(H_optimal * (sequence_count[None] - np.eye(n_sequences)), axis=1)
                
                # assign infinite distance to invalid sequences 
                # i.e., sequences for which the number of contexts is not equal to the most common number of contexts
                H_mean[sequence_count == 0] = np.inf
                
                # find the index of the typical sequence 
                # (the sequence with minimum mean optimal Hamming distance to all other sequences)
                min_ind = np.argmin(H_mean, axis=0)
                
                # typical context sequence
                typical_sequence = unique_seqs[min_ind, :]
                
                # store the optimal permutation of labels for each sequence with respect to the typical sequence
                j = np.argmin(H[min_ind, :, :], axis=-1)
                optimal_assignment[i] = np.transpose(
                    L[:int(mode_number_of_contexts[i]), :, j].reshape((int(mode_number_of_contexts[i]), -1, 1)), 
                    [2, 1, 0], 
                )[0]
        
        return P, S, optimal_assignment, from_unique, context_sequence, C
        
    def resample_inds(self, S: Dict[str, Any]):
        num_trials = len(self.perturbations)
        
        inds_resampled = np.zeros((self.particles * self.runs, num_trials), dtype=int)
        
        for n in range(self.runs):
            p = self.particles * n + np.arange(self.particles)
            inds_resampled[p, :] = self.particles * n + S["runs"][n]["inds_resampled"]
        
        return inds_resampled
    
    def context_sequence(self, S: Dict[str, Any], inds_resampled: np.ndarray):
        num_trials = len(self.perturbations)
        
        context_seq = {}
        
        for n in range(self.runs):
            p = self.particles * n + np.arange(self.particles)
            for i in range(num_trials):
                if n == 0:
                    context_seq[i] = np.zeros((self.particles * self.runs, i+1), dtype=int)
                if i > 0:
                    context_seq[i][p, :i] = context_seq[i-1][p, :]
                    context_seq[i][p, :] = context_seq[i][inds_resampled[p, i], :]
                context_seq[i][p, i] = S["runs"][n]["context"][:, i]
                
        return context_seq
    
    def posterior_number_of_contexts(self, context_sequence: Dict[int, Any], S: Dict[str, Any]):
        num_trials = len(self.perturbations)
        
        # number of contexts
        C = np.zeros((self.particles * self.runs, num_trials), dtype=int)
        for n in range(self.runs):
            p = self.particles * n + np.arange(self.particles)
            for i in range(num_trials):
                C[p, i] = np.max(context_sequence[i][p, :], axis=1)
        
        particle_weight = np.repeat(S["weights"], self.particles) / self.particles
        
        posterior = np.zeros((self.max_contexts+1, num_trials))
        posterior_mean = np.zeros((num_trials, ))
        posterior_mode = np.zeros((num_trials, ))
        
        for i in range(num_trials):
            for context in range(np.max(C[:, i])):
                posterior[context, i] = np.sum((C[:, i] == (context+1)) * particle_weight)
            
            posterior_mean[i] = np.sum((np.arange(self.max_contexts+1) + 1) * posterior[:, i])
            posterior_mode[i] = np.argmax(posterior[:, i]) + 1
        
        return C, posterior, posterior_mean, posterior_mode

    def compute_variables_for_plotting(
        self, 
        P: Dict[str, Any], 
        S: Dict[str, Any], 
        optimal_assignment: Dict[int, Any], 
        from_unique: Dict[int, Any], 
        context_sequence: Dict[int, Any], 
        C: np.ndarray, 
    ):
        num_trials = len(self.perturbations)
        
        P = self.preallocate_memory(P)
        
        n_particles_used = np.zeros((num_trials, self.runs))
        for i in range(num_trials):
            if np.mod(i+1, 50) == 0:
                print(f"Permuting context labels (trial = {i+1})")
            
            # cumulative number of particles for which C <= np.max(P["mode_number_of_contexts"])
            N = 0
            
            for n in range(self.runs):
                # inds of particles of the current run
                p = self.particles * n + np.arange(self.particles)

                # inds of particles that are either valid now or could be valid in the future
                # C <= np.max(P["mode_number_of_contexts"])
                valid_future = np.where(C[p, i] <= np.max(P["mode_number_of_contexts"]))[0]
                
                # inds of particles that are valid now
                # C == np.max(P["mode_number_of_contexts"])
                valid_now = np.where(C[p, i] == P["mode_number_of_contexts"][i])[0]
                n_particles_used[i, n] = len(valid_now)
                
                if len(valid_now) > 0:
                    for particle in valid_now:
                        # index of the optimal label permutations of the current particle
                        ind = N + np.where(particle == valid_future)[0]
                        
                        # is the latest context a novel context
                        # this is needed to store novel context probabilities
                        context_trajectory = context_sequence[i][p[particle], :]
                        try:
                            novel_context = context_trajectory[i] > np.max(context_trajectory[:i])
                        except Exception as e:
                            novel_context = False
                        
                        S = self.relabel_context_variables(S, optimal_assignment[i][from_unique[i][ind], :].astype(int)-1, novel_context, particle, i, n)
                    P = self.integrate_over_particles(S, P, valid_now, i, n)
                
                N += len(valid_future)
        
        P = self.integrate_over_runs(P, S)
        P = self.normalise_relabelled_variables(P, n_particles_used, S)
        
        if self.plot_state_given_context:
            P["state_given_novel_context"] = np.tile(
                np.nanmean(P["state_given_context"][:, :, -1], axis=1, keepdims=True), 
                [1, num_trials, 1], 
            )
            P["state_given_context"] = P["state_given_context"][:, :, :-1]
        
        return P, S
        
    def preallocate_memory(self, P: Dict[str, Any]):
        num_trials = len(self.perturbations)

        mode_number_of_contexts = P["mode_number_of_contexts"].astype(int)
        if self.plot_state_given_context:
            P["state_given_context"] = np.ones(
                (self.state_values.size, num_trials, np.max(mode_number_of_contexts)+1, self.runs)
            ) * np.nan
        if self.plot_predicted_probabilities:
            P["predicted_probabilities"] = np.ones((num_trials, np.max(mode_number_of_contexts)+1, self.runs)) * np.nan
            P["predicted_probabilities"][0, -1, :] = 1
        if self.plot_responsibilities:
            P["responsibilities"] = np.ones((num_trials, np.max(mode_number_of_contexts)+1, self.runs)) * np.nan
        if self.plot_stationary_probabilities:
            P["stationary_probabilities"] = np.ones((num_trials, np.max(mode_number_of_contexts)+1, self.runs)) * np.nan
        if self.plot_retention_given_context:
            P["retention_given_context"] = np.zeros(
                (self.retention_values.size, num_trials, np.max(mode_number_of_contexts), self.runs)
            )
        if self.plot_drift_given_context:
            P["drift_given_context"] = np.zeros(
                (self.drift_values.size, num_trials, np.max(mode_number_of_contexts), self.runs)
            )
        if self.plot_bias_given_context:
            P["bias_given_context"] = np.zeros(
                (self.bias_values.size, num_trials, np.max(mode_number_of_contexts), self.runs)
            )
        if self.plot_global_transition_probabilities:
            P["global_transition_probabilities"] = np.ones((num_trials, np.max(mode_number_of_contexts)+1, self.runs)) * np.nan
        if self.plot_local_transition_probabilities:
            P["local_transition_probabilities"] = np.ones(
                (np.max(mode_number_of_contexts), np.max(mode_number_of_contexts)+1, num_trials, self.runs)
            ) * np.nan
        if self.plot_global_cue_probabilities:
            P["global_cue_probabilities"] = np.ones((num_trials, np.max(self.cues)+1, self.runs)) * np.nan
        if self.plot_local_cue_probabilities:
            P["local_cue_probabilities"] = np.ones(
                (np.max(mode_number_of_contexts), np.max(self.cues)+1, num_trials, self.runs)
            ) * np.nan
        if self.plot_state:
            P["state"] = np.ones((self.state_values.size, num_trials, self.runs)) * np.nan
        if self.plot_average_state or self.plot_state:
            P["average_state"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_bias:
            P["bias"] = np.ones((self.bias_values.size, num_trials, self.runs)) * np.nan
        if self.plot_average_bias or self.plot_bias:
            P["average_bias"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_state_feedback:
            P["state_feedback"] = np.ones((self.state_feedback_values.size, num_trials, self.runs)) * np.nan
        if self.plot_explicit_component:
            P["explicit_component"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_implicit_component:
            P["implicit_component"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_Kalman_gain_given_cstar1:
            P["Kalman_gain_given_cstar1"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_predicted_probability_cstar1:
            P["predicted_probability_cstar1"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_state_given_cstar1:
            P["state_given_cstar1"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_Kalman_gain_given_cstar2:
            P["Kalman_gain_given_cstar2"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_state_given_cstar2:
            P["state_given_cstar2"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_predicted_probability_cstar3:
            P["predicted_probability_cstar3"] = np.ones((num_trials, self.runs)) * np.nan
        if self.plot_state_given_cstar3:
            P["state_given_cstar3"] = np.ones((num_trials, self.runs)) * np.nan
        P["average_state_feedback"] = np.ones((num_trials, self.runs)) * np.nan
    
        return P

    def relabel_context_variables(
        self, 
        S: Dict[str, Any], 
        optimal_assignment: np.ndarray, 
        novel_context: bool, 
        particle: int, 
        trial: int, 
        run: int, 
    ):
        C = optimal_assignment.size
        num_trials = len(self.perturbations)
        
        # predictive distribution
        if trial < (num_trials-1):
            if self.plot_state_given_context:
                S["runs"][run]["state_mean"][optimal_assignment, particle, trial+1] = S["runs"][run]["state_mean"][:C, particle, trial+1]
                S["runs"][run]["state_var"][optimal_assignment, particle, trial+1] = S["runs"][run]["state_var"][:C, particle, trial+1]
            if self.plot_predicted_probabilities:
                S["runs"][run]["predicted_probabilities"][optimal_assignment, particle, trial+1] = S["runs"][run]["predicted_probabilities"][:C, particle, trial+1]
        
        if self.plot_responsibilities:
            if (trial == 0) or novel_context:
                S["runs"][run]["responsibilities"][np.concatenate([np.arange(C-1), np.array([C])]), particle, trial] = \
                    S["runs"][run]["responsibilities"][optimal_assignment, particle, trial]
                S["runs"][run]["responsibilities"][C-1, particle, trial] = np.nan
            else:
                S["runs"][run]["responsibilities"][np.arange(C+1), particle, trial] = \
                    S["runs"][run]["responsibilities"][np.concatenate([optimal_assignment, np.array([C])]), particle, trial]
        if self.plot_stationary_probabilities:
            S["runs"][run]["stationary_probabilities"][optimal_assignment, particle, trial] = S["runs"][run]["stationary_probabilities"][:C, particle, trial]
        if self.plot_retention_given_context or self.plot_drift_given_context:
            S["runs"][run]["dynamics_mean"][:, optimal_assignment, particle, trial] = S["runs"][run]["dynamics_mean"][:, :C, particle, trial]
            S["runs"][run]["dynamics_var"][:, optimal_assignment, particle, trial] = S["runs"][run]["dynamics_var"][:, :C, particle, trial]
        if self.plot_bias_given_context:
            S["runs"][run]["bias_mean"][optimal_assignment, particle, trial] = S["runs"][run]["bias_mean"][:C, particle, trial]
            S["runs"][run]["bias_var"][optimal_assignment, particle, trial] = S["runs"][run]["bias_var"][:C, particle, trial]
        if self.plot_global_transition_probabilities:
            S["runs"][run]["global_transition_posterior"][optimal_assignment, particle, trial] = S["runs"][run]["global_transition_posterior"][:C, particle, trial]
        if self.plot_local_transition_probabilities:
            S["runs"][run]["local_transition_matrix"][:C, :(C+1), particle, trial] = \
                self.permute_transition_matrix_columns_and_rows(S["runs"][run]["local_transition_matrix"][:C, :(C+1), particle, trial], optimal_assignment)
        if self.plot_local_cue_probabilities:
            S["runs"][run]["local_cue_matrix"][optimal_assignment, :, particle, trial] = S["runs"][run]["local_cue_matrix"][:C, :, particle, trial]
        
        return S
    
    def permute_transition_matrix_columns_and_rows(self, transmat: np.ndarray, optimal_assignment: np.ndarray):
        C = len(optimal_assignment)
        # inverse mapping (TODO: verify the dimensions == C)
        inds_map, _ = np.where(optimal_assignment == np.arange(C)[None])
        # inds_map = np.equal(optimal_assignment[:, None], np.arange(C)[None])
        
        inds_1 = np.tile(inds_map[:, None], [1, C+1])
        inds_2 = np.tile(np.concatenate([inds_map, np.array([C+1])])[None], [C, 1])
        
        permuted_transmat = np.zeros_like(transmat)
        for i in range(C):
            for j in range(C+1):
                permuted_transmat[i, j] = transmat[inds_1[i], inds_2[j]]
        
        return permuted_transmat
    
    def integrate_over_particles(
        self, 
        S: Dict[str, Any], 
        P: Dict[str, Any], 
        particles: np.ndarray, 
        trial: int, 
        run: int, 
    ):
        C = int(P["mode_number_of_contexts"][trial])
        novel_context = int(np.max(P["mode_number_of_contexts"])) + 1
        
        num_trials = len(self.perturbations)
        
        # predictive distributions
        if trial < (num_trials - 1):
            if self.plot_state_given_context:
                mu = np.transpose(S["runs"][run]["state_mean"][:(C+1), particles, trial+1][..., None], [2, 1, 0])
                sd = np.transpose(np.sqrt(S["runs"][run]["state_var"][:(C+1), particles, trial+1][..., None]), [2, 1, 0])
                # TODO: check the dimensions!
                P["state_given_context"][:, trial+1, np.concatenate([np.arange(C), np.array([novel_context])-1]), run] = \
                    np.sum(norm(mu, sd).pdf(self.state_values[:, None, None]), axis=1)
            if self.plot_predicted_probabilities:
                P["predicted_probabilities"][trial+1, np.concatenate([np.arange(C), np.array([novel_context])-1]), run] = \
                    np.sum(S["runs"][run]["predicted_probabilities"][:(C+1), particles, trial+1], axis=1)
        
        if self.plot_responsibilities:
            P["responsibilities"][trial, np.concatenate([np.arange(C), np.array([novel_context])-1]), run] = \
                self.sum_along_dimension(S["runs"][run]["responsibilities"][:(C+1), particles, trial], axis=1)
        if self.plot_stationary_probabilities:
            P["stationary_probabilities"][trial, np.concatenate([np.arange(C), np.array([novel_context])-1]), run] = \
                np.sum(S["runs"][run]["stationary_probabilities"][:(C+1), particles, trial], axis=1, keepdims=True)
        if self.plot_retention_given_context:
            mu = np.transpose(S["runs"][run]["dynamics_mean"][0, :C, particles, trial], [1, 0])
            std = np.transpose(np.sqrt(S["runs"][run]["dynamics_mean"][0, 0, :C, particles, trial]), [1, 0])
            P["retention_given_context"][:, trial, :C, run] = np.sum(norm(mu, std).pdf(self.retention_values), axis=1, keepdims=True)
        if self.plot_drift_given_context:
            mu = np.transpose(S["runs"][run]["dynamics_mean"][1, :C, particles, trial], [1, 0])
            std = np.transpose(np.sqrt(S["runs"][run]["dynamics_mean"][1, 1, :C, particles, trial]), [1, 0])
            P["drift_given_context"][:, trial, :C, run] = np.sum(norm(mu, std).pdf(self.drift_values), axis=1, keepdims=True)
        if self.plot_bias_given_context:
            mu = np.transpose(S["runs"][run]["bias_mean"][:C, particles, trial], [1, 0])
            std = np.transpose(np.sqrt(S["runs"][run]["bias_var"][:C, particles, trial]), [1, 0])
            P["bias_given_context"][:, trial, :C, run] = np.sum(norm(mu, std).pdf(self.bias_values), axis=1, keepdims=True)
        if self.plot_global_transition_probabilities:
            alpha = S["runs"][run]["global_transition_posterior"][:(C+1), particles, trial]
            P["global_transition_probabilities"][trial, np.concatenate([np.arange(C), np.array([novel_context])]), run] = \
                np.sum(alpha / np.sum(alpha, axis=0, keepdims=True), axis=1, keepdims=True)
        if self.plot_local_transition_probabilities:
            P["local_transition_probabilities"][:C, np.concatenate([np.arange(C), np.array([novel_context])]), trial, run] = \
                np.sum(S["runs"][run]["local_transition_matrix"][:C, :(C+1), particles, trial], axis=2, keepdims=True)
        if self.plot_local_cue_probabilities:
            P["local_cue_probabilities"][
                :C, np.concatenate([np.arange(np.max(self.cues[:trial])), np.array([np.max(self.cues)+1])]), trial, run
            ] = np.sum(S["runs"][run]["local_cue_matrix"][:C, :(np.max(self.cues[:trial])+1), particles, trial], axis=2, keepdims=True)
        
        return P
    
    def sum_along_dimension(self, X: np.ndarray, dim: int):
        nan_inds = np.all(np.isnan(X), axis=dim)
        X = np.nansum(X, axis=dim)
        X[nan_inds] = np.nan
        
        return X
    
    def integrate_over_runs(self, P: Dict[str, Any], S: Dict[str, Any]):
        if self.plot_state_given_context:
            P["state_given_context"] = self.weighted_sum_along_dimension(P["state_given_context"], S, dim=3)
        if self.plot_predicted_probabilities:
            P["predicted_probabilities"] = self.weighted_sum_along_dimension(P["predicted_probabilities"], S, dim=2)
        if self.plot_responsibilities:
            P["responsibilities"] = self.weighted_sum_along_dimension(P["responsibilities"], S, dim=2)
        if self.plot_stationary_probabilities:
            P["stationary_probabilities"] = self.weighted_sum_along_dimension(P["stationary_probabilities"], S, dim=2)
        if self.plot_retention_given_context:
            P["retention_given_context"] = self.weighted_sum_along_dimension(P["retention_given_context"], S, dim=3)
        if self.plot_drift_given_context:
            P["drift_given_context"] = self.weighted_sum_along_dimension(P["drift_given_context"], S, dim=3)
        if self.plot_bias_given_context:
            P["bias_given_context"] = self.weighted_sum_along_dimension(P["bias_given_context"], S, dim=3)
        if self.plot_global_transition_probabilities:
            P["global_transition_probabilities"]=  self.weighted_sum_along_dimension(
                P["global_transition_probabilities"], S, dim=2, 
            )
        if self.plot_local_transition_probabilities:
            P["local_transition_probabilities"]=  self.weighted_sum_along_dimension(
                P["local_transition_probabilities"], S, dim=3, 
            )
        if self.plot_global_cue_probabilities:
            for n in range(self.runs):
                for i in range(len(self.perturbations)):
                    alpha = S["runs"][n]["global_cue_posterior"][:(np.max(self.cues[:i])+1), :, i]
                    P["global_cue_probabilities"][i, np.concatenate([np.arange(np.max(self.cues[:i])), np.array([np.max(self.cues)+1])]), n] = \
                        np.sum(alpha / np.sum(alpha, axis=0, keepdims=True), axis=1)
            P["global_cue_probabilities"] = self.weighted_sum_along_dimension(P["global_cue_probabilities"], S, axis=2)
        if self.plot_local_cue_probabilities:
            P["local_cue_probabilities"] = self.weighted_sum_along_dimension(P["local_cue_probabilities"], S, axis=3)
        
        if self.plot_state:
            for n in range(self.runs):
                P["state"][:, :, n] = S["runs"][n]["state_distribution"]
            P["state"] = self.weighted_sum_along_dimension(P["state"], S, dim=2)
        if self.plot_average_state or self.plot_state:
            for n in range(self.runs):
                P["average_state"][:, n] = S["runs"][n]["average_state"]
            P["average_state"] = self.weighted_sum_along_dimension(P["average_state"], S, dim=1)
        if self.plot_bias:
            for n in range(self.runs):
                P["bias"][:, :, n] = S["runs"][n]["bias_distribution"]
            P["bias"] = self.weighted_sum_along_dimension(P["bias"], S, dim=2)
        if self.plot_average_bias or self.plot_bias:
            for n in range(self.runs):
                P["average_bias"][:, n] = S["runs"][n]["implicit"]
            P["average_bias"] = self.weighted_sum_along_dimension(P["average_bias"], S, dim=1)
        if self.plot_state_feedback:
            for n in range(self.runs):
                P["state_feedback"][:, :, n] = S["runs"][n]["state_feedback_distribution"]
            P["state_feedback"] = self.weighted_sum_along_dimension(P["state_feedback"], S, dim=2)
        if self.plot_explicit_component:
            for n in range(self.runs):
                P["explicit_component"][:, n]=  S["runs"][n]["explicit"]
            P["explicit_component"] = self.weighted_sum_along_dimension(P["explicit_component"], S, dim=1)
        if self.plot_implicit_component:
            for n in range(self.runs):
                P["implicit_component"][:, n]=  S["runs"][n]["implicit"]
            P["implicit_component"] = self.weighted_sum_along_dimension(P["implicit_component"], S, dim=1)
        if self.plot_Kalman_gain_given_cstar1:
            for n in range(self.runs):
                P["Kalman_gain_given_cstar1"][:, n] = S["runs"][n]["Kalman_gain_given_cstar1"]
            P["Kalman_gain_given_cstar1"] = self.weighted_sum_along_dimension(P["Kalman_gain_given_cstar1"], S, axis=1)
        if self.plot_predicted_probability_cstar1:
            for n in range(self.runs):
                P["predicted_probability_cstar1"][:, n] = S["runs"][n]["predicted_probability_cstar1"]
            P["predicted_probability_cstar1"] = self.weighted_sum_along_dimension(P["predicted_probability_cstar1"], S, dim=1)
        if self.plot_state_given_cstar1:
            for n in range(self.runs):
                P["state_given_cstar1"][:, n] = S["runs"][n]["state_given_cstar1"]
            P["state_given_cstar1"] = self.weighted_sum_along_dimension(P["state_given_cstar1"], S, dim=1)
        if self.plot_Kalman_gain_given_cstar2:
            for n in range(self.runs):
                P["Kalman_gain_given_cstar2"][:, n] = S["runs"][n]["Kalman_gain_given_cstar2"]
            P["Kalman_gain_given_cstar2"] = self.weighted_sum_along_dimension(P["Kalman_gain_given_cstar2"], S, axis=1)
        if self.plot_state_given_cstar2:
            for n in range(self.runs):
                P["state_given_cstar2"][:, n] = S["runs"][n]["state_given_cstar2"]
            P["state_given_cstar2"] = self.weighted_sum_along_dimension(P["state_given_cstar2"], S, dim=1)
        if self.plot_predicted_probability_cstar3:
            for n in range(self.runs):
                P["predicted_probability_cstar3"][:, n] = S["runs"][n]["predicted_probability_cstar3"]
            P["predicted_probability_cstar3"] = self.weighted_sum_along_dimension(P["predicted_probability_cstar3"], S, dim=1)
        if self.plot_state_given_cstar3:
            for n in range(self.runs):
                P["state_given_cstar3"][:, n] = S["runs"][n]["state_given_cstar3"]
            P["state_given_cstar3"] = self.weighted_sum_along_dimension(P["state_given_cstar3"], S, dim=1)
        for n in range(self.runs):
            P["average_state_feedback"][:, n] = S["runs"][n]["motor_output"]
        P["average_state_feedback"] = self.weighted_sum_along_dimension(P["average_state_feedback"], S, dim=1)

        return P
    
    def weighted_sum_along_dimension(self, X: np.ndarray, S: Dict[str, Any], dim: int):
        nan_inds = np.all(np.isnan(X), axis=dim)
        X = np.nansum(X * np.reshape(S["weights"], [1] * (dim-1) + [S["weights"].size]), axis=dim)
        X[nan_inds] = np.nan
        
        return X
    
    def normalise_relabelled_variables(
        self, 
        P: Dict[str, Any], 
        n_particles_used: np.ndarray, 
        S: Dict[str, Any], 
    ):
        # normalisation constant
        Z = np.sum(n_particles_used * S["weights"], axis=1, keepdims=True)
        
        if self.plot_state_given_context:
            P["state_given_context"][:, 1:, :] = P["state_given_context"][:, 1:, :] / Z[:-1][None]
        if self.plot_predicted_probabilities:
            P["predicted_probabilities"][1:, :] = P["predicted_probabilities"][1:, :] / Z[:-1]
        if self.plot_responsibilities:
            P["responsibilities"] = P["responsibilities"] / Z[None]
            P["novel_context_responsibility"] = P["responsibilities"][:, -1]
            P["known_context_responsibilities"] = P["responsibilities"][:, :-1]
        if self.plot_stationary_probabilities:
            P["stationary_probabilities"] = P["stationary_probabilities"] / Z[None]
        if self.plot_retention_given_context:
            P["retention_given_context"] = P["retention_given_context"] / Z[:, None]
        if self.plot_drift_given_context:
            P["drift_given_context"] = P["drift_given_context"] / Z[:, None]
        if self.plot_bias_given_context:
            P["bias_given_context"] = P["bias_given_context"] / Z[:, None]
        if self.plot_global_transition_probabilities:
            P["global_transition_probabilities"] = P["global_transition_probabilities"] / Z[None]
        if self.plot_local_transition_probabilities:
            P["local_transition_probabilities"] = P["local_transition_probabilities"] / Z[None, None]
        if self.plot_global_cue_probabilities:
            P["global_cue_probabilities"] = P["global_cue_probabilities"] / self.particles
        if self.plot_local_cue_probabilities:
            P["local_cue_probabilities"] = P["local_cue_probabilities"] / Z[None, None]

        return P

    def generate_figures(self, P: Dict[str, Any]):
        colors = self.colors()
        
        lw = 2
        font_size = 15
        
        num_trials = len(self.perturbations)
        
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        
        if self.plot_state_given_context:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            y_lims = [self.state_values[0], self.state_values[-1]]
            y_ticks = [-1.0, 0.0, 1.0]
            self.plot_image(
                P["state_given_context"], 
                y_lims, 
                y_ticks, 
                colors["contexts"][:np.max(P["mode_number_of_contexts"]).astype(int), :], 
                ax=ax[0], 
            )
            ax[0].set_xticks([0, num_trials-1])
            ax[0].set_xticklabels([1, num_trials])
            ax[0].set_xlim([0, num_trials-1])
            
            ax[0].set_ylabel("state|context")
            ax[0].set_xlabel("trials")
            
            self.plot_image(P["state_given_novel_context"], y_lims, y_ticks, colors["new_context"], ax=ax[1])
            ax[1].set_xticks([0, num_trials-1])
            ax[1].set_xticklabels([1, num_trials])
            ax[1].set_xlim([0, num_trials-1])
            
            ax[1].set_ylabel("state|novel_context")
            ax[1].set_xlabel("trials")
            
            fig.savefig(os.path.join(self.fig_dir, "state_given_contexts.png"))
        
        if self.plot_predicted_probabilities:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["predicted_probabilities"][:, -1], color=colors["new_context"][0], linewidth=lw)
            for context in range(int(np.max(P["mode_number_of_contexts"]))):
                predicted_probs = P["predicted_probabilities"][:, context]
                t = np.where(~np.isnan(P["predicted_probabilities"][:, context]))[0][0]
                predicted_probs[t-1] = P["predicted_probabilities"][t-1, -1]
                ax.plot(predicted_probs, color=colors["contexts"][context, :], linewidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("predicted probability")
            
            fig.savefig(os.path.join(self.fig_dir, "predicted_probabilities.png"))
        
        if self.plot_responsibilities:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            for context in range(np.max(P["mode_number_of_contexts"])):
                ax[0].plot(P["known_context_responsibilties"][:, context], color=colors["contexts"][context, :], linewidth=lw)
            ax[0].set_xlim(0, num_trials)
            ax[0].set_ylim(-0.1, 1.1)
            ax[0].set_xticks([0, num_trials])
            ax[0].set_yticks([0, 1])
            ax[0].set_xlabel("trials")
            ax[0].set_ylabel("known context responsibility")
            
            ax[1].plot(P["novel_context_responsibility"], color=colors["new_context"][0], linewidth=lw)
            ax[1].set_xlim(0, num_trials)
            ax[1].set_ylim(-0.1, 1.1)
            ax[1].set_xticks([0, num_trials])
            ax[1].set_yticks([0, 1])
            ax[1].set_xlabel("trials")
            ax[1].set_ylabel("novel context responsibility")
            
            fig.savefig(os.path.join(self.fig_dir, "responsibilities.png"))
            
        if self.plot_stationary_probabilities:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["stationary_probabilities"][:, -1], color=colors["new_context"][0], linewidth=lw)
            for context in range(np.max(P["mode_number_of_contexts"])):
                ax.plot(P["stationary_probabilities"][:, context], color=colors["contexts"][context, :], linewidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([0.0, 1.0])
            ax.set_xlabel("trials")
            ax.set_ylabel("stationary context probability")
            
            fig.savefig(os.path.join(self.fig_dir, "stationary_probabilities.png"))
        
        if self.plot_retention_given_context:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            y_lims = [self.retention_values[0], self.retention_values[-1]]
            y_ticks = [0, self.retention_values[0], self.retention_values[-1]]
            self.plot_image(P["retention_given_context"], y_lims, y_ticks, colors["contexts"], ax=ax)
            ax.set_xticks([0, num_trials-1])
            ax.set_xticklabels([1, num_trials])
            ax.set_xlim([0, num_trials-1])
            ax.set_xlabel("trials")
            ax.set_ylabel("retention|context")
            
            fig.savefig(os.path.join(self.fig_dir, "retention.png"))
        
        if self.plot_drift_given_context:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            y_lims = [self.drift_values[0], self.drift_values[-1]]
            y_ticks = [0, self.drift_values[0], self.drift_values[-1]]
            self.plot_image(P["drift_given_context"], y_lims, y_ticks, colors["contexts"], ax=ax)
            ax.set_xticks([0, num_trials-1])
            ax.set_xticklabels([1, num_trials])
            ax.set_xlim([0, num_trials-1])
            ax.set_xlabel("trials")
            ax.set_ylabel("drift|context")
            
            fig.savefig(os.path.join(self.fig_dir, "drift.png"))
        
        if self.plot_bias_given_context:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            y_lims = [self.bias_values[0], self.bias_values[-1]]
            y_ticks = [0, self.bias_values[0], self.bias_values[-1]]
            self.plot_image(P["bias_given_context"], y_lims, y_ticks, colors["contexts"], ax=ax)
            ax.set_xticks([0, num_trials-1])
            ax.set_xticklabels([1, num_trials])
            ax.set_xlim([0, num_trials-1])
            ax.set_xlabel("trials")
            ax.set_ylabel("bias|context")
            
            fig.savefig(os.path.join(self.fig_dir, "bias.png"))
        
        if self.plot_global_transition_probabilities:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["global_transition_probabilities"][:, -1], color=colors["new_context"][0], linewidth=lw)
            for context in range(np.max(P["mode_number_of_contexts"])):
                ax.plot(P["global_transition_probabilities"][:, context], color=colors["contexts"][context, :], linewidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("global transition probability")
            
            fig.savefig(os.path.join(self.fig_dir, "global_transition_probabilities.png"))
        
        if self.plot_local_transition_probabilities:
            fig, ax = plt.subplots(
                1, np.max(P["mode_number_of_contexts"]), figsize=(6*np.max(P["mode_number_of_contexts"]), 5), 
            )
            for from_context in range(np.max(P["mode_number_of_contexts"])):
                ax[from_context].plot(
                    P["local_transition_probabilities"][from_context, -1, :], color=colors["new_context"][0], linewidth=lw, label="to new context"
                )
                for to_context in range(np.max(P["mode_number_of_contexts"])):
                    ax[from_context].plot(
                        P["local_transition_probabilities"][from_context, to_context, :], color=colors["contexts"][context, :], linewidth=lw, label=f"to context {to_context}", 
                    )
                ax[from_context].set_title(f"from context {from_context}")
                ax[from_context].set_xlim(0, num_trials)
                ax[from_context].set_ylim(-0.1, 1.1)
                ax[from_context].set_xtick([0, num_trials])
                ax[from_context].set_ytick([0, 1])
                ax[from_context].set_xlabel("trials")
                ax[from_context].set_ylabel("local transition probability")
                ax[from_context].legend()
                
            fig.savefig(os.path.join(self.fig_dir, "local_transition_probabilities.png"))
                
        if self.plot_global_cue_probabilities:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["global_cue_probabilities"][:, -1], color=colors["new_context"][0], linewidth=lw, label="novel cue")
            for cue in range(np.max(self.cues)):
                ax.plot(P["global_cue_probabilities"][:, cue], color=colors["cues"][cue, :], linewidth=lw, label=f"cue {cue}")
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("global cue probability")
            ax.legend()
            
            fig.savefig(os.path.join(self.fig_dir, "global_cue_probabilities.png"))
        
        if self.plot_local_cue_probabilities:
            fig, ax = plt.subplots(
                1, np.max(P["mode_number_of_contexts"]), figsize=(6*np.max(P["mode_number_of_contexts"]), 5), 
            )
            for context in range(np.max(P["mode_number_of_contexts"])):
                ax[context].plot(P["local_cue_probabilities"][context, -1, :], color=colors["new_context"][0], linewidth=lw, label="novel context")
                for cue in range(np.max(self.cues)):
                    ax[context].plot(P["local_cue_probabilities"][context, cue, :], color=colors["cues"][cue, :], linewidth=lw, label=f"cue {cue}")
                ax[context].set_title(f"context {context}")
                ax[context].set_xlim(0, num_trials)
                ax[context].set_ylim(-0.1, 1.1)
                ax[context].set_xticks([0, num_trials])
                ax[context].set_yticks([0, 1])
                ax[context].set_xlabel("trials")
                ax[context].set_ylabel("local cue probability")
                ax[context].legend()
            
            fig.savefig(os.path.join(self.fig_dir, "local_cue_probabilities.png"))
        
        if self.plot_state:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            y_lims = [self.state_values[0], self.state_values[-1]]
            y_ticks = [-1, 0, 1]
            self.plot_image(P["state"][..., None], y_lims, y_ticks, colors["marginal"], ax=ax)
            n_pixels = len(self.state_values)
            ax.plot(self.map_to_pixel_space(n_pixels, y_lims, P["average_state"]), color=colors["mean_of_marginal"], linewidth=lw)
            ax.set_xticks([0, num_trials-1])
            ax.set_xticklabels([1, num_trials])
            ax.set_xlim(0, num_trials-1)
            ax.set_xlabel("trials")
            ax.set_ylabel("state")
            
            fig.savefig(os.path.join(self.fig_dir, "state.png"))
        
        if self.plot_average_state:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["average_state"], color=colors["mean_of_marginal"], linewidth=lw)
            ax.set_xlim([0, num_trials])
            ax.set_ylim([-1.5, 1.5])
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("average state")
            
            fig.savefig(os.path.join(self.fig_dir, "average_state.png"))
        
        if self.plot_bias:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            y_lims = [self.bias_values[0], self.bias_values[-1]]
            y_ticks = [-1, 0, 1]
            self.plot_image(P["bias"], y_lims, y_ticks, colors["marginal"], ax=ax)
            n_pixels = len(self.bias_values)
            ax.plot(self.map_to_pixel_space(n_pixels, y_lims, P["average_bias"]), color=colors["mean_of_marginal"], linewidth=lw)
            ax.set_xticks([0, num_trials-1])
            ax.set_xticklabels([1, num_trials])
            ax.set_xlim(0, num_trials-1)
            ax.set_xlabel("trials")
            ax.set_ylabel("bias")
            
            fig.savefig(os.path.join(self.fig_dir, "bias.png"))
        
        if self.plot_average_bias:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["average_bias"], color=colors["mean_of_marginal"], linewidth=lw)
            ax.set_xlim(0, num_trials-1)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([0, num_trials-1])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("average bias")
            
            fig.savefig(os.path.join(self.fig_dir, "average_bias.png"))
        
        if self.plot_state_feedback:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            y_lims = [self.state_feedback_values[0], self.state_feedback_values[-1]]
            y_ticks = [-1, 0, 1]
            self.plot_image(P["state_feedback"], y_lims, y_ticks, colors["marginal"], ax=ax)
            n_pixels = len(self.state_feedback_values)
            ax.plot(self.map_to_pixel_space(n_pixels, y_lims, P["average_state_feedback"]), color=colors["mean_of_marginal"], linewidth=lw)
            ax.set_xticks([0, num_trials-1])
            ax.set_xticklabels([1, num_trials])
            ax.set_xlim([0, num_trials-1])
            ax.set_xlabel("trials")
            ax.set_ylabel("state feedback")
            
            fig.savefig(os.path.join(self.fig_dir, "state_feedback.png"))
            
        if self.plot_explicit_component:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["explicit_component"], color=colors["mean_of_marginal"], linewidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("explicit component fo adaptation")
            
            fig.savefig(os.path.join(self.fig_dir, "explicit_component.png"))
        
        if self.plot_implicit_component:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["implicit_component"], color=colors["mean_of_marginal"], linewidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("implicit component fo adaptation")
            
            fig.savefig(os.path.join(self.fig_dir, "implicit_component.png"))
        
        if self.plot_Kalman_gain_given_cstar1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["Kalman_gain_given_cstar1"], color="k", linwidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("Kalman_gain|c^*")
            ax.set_title("c^* is the context with the highest responsibility")
            
            fig.savefig(os.path.join(self.fig_dir, "Kalman_gain_given_cstar1.png"))
        
        if self.plot_predicted_probability_cstar1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["predicted_probability_cstar1"], color="k", linwidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("predicted probability (c^*)")
            ax.set_title("c^* is the context with the highest responsibility")
            
            fig.savefig(os.path.join(self.fig_dir, "predicted_probability_cstar1.png"))
        
        if self.plot_state_given_cstar1:
            ax.plot(P["state_given_cstar1"], color="k", linwidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("E[state|c^*]")
            ax.set_title("c^* is the context with the highest responsibility")
            
            fig.savefig(os.path.join(self.fig_dir, "state_given_cstar1.png"))
            
        if self.plot_Kalman_gain_given_cstar2:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["Kalman_gain_given_cstar2"], color="k", linwidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("Kalman_gain|c^*")
            ax.set_title("c^* is the context with the highest predicted probability on the next trial")
            
            fig.savefig(os.path.join(self.fig_dir, "Kalman_gain_given_cstar2.png"))
        
        if self.plot_state_given_cstar2:
            ax.plot(P["state_given_cstar2"], color="k", linwidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("E[state|c^*]")
            ax.set_title("c^* is the context with the highest predicted probability on the next trial")
            
            fig.savefig(os.path.join(self.fig_dir, "state_given_cstar2.png"))
        
        if self.plot_predicted_probability_cstar3:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(P["predicted_probability_cstar3"], color="k", linwidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("predicted probability (c^*)")
            ax.set_title("c^* is the context with the highest predicted probability")
            
            fig.savefig(os.path.join(self.fig_dir, "predicted_probability_cstar3.png"))
        
        if self.plot_state_given_cstar3:
            ax.plot(P["state_given_cstar3"], color="k", linwidth=lw)
            ax.set_xlim(0, num_trials)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([0, num_trials])
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel("trials")
            ax.set_ylabel("E[state|c^*]")
            ax.set_title("c^* is the context with the highest predicted probability")
            
            fig.savefig(os.path.join(self.fig_dir, "state_given_cstar3.png"))
    
    def plot_image(
        self, 
        X: np.ndarray, 
        y_lim: List[float], 
        y_ticks: List[float], 
        rgb_colors: List[List[float]], 
        ax=None
    ):
        if ax is None:
            ax = plt.gca()
        data = np.zeros((X.shape[0], X.shape[1], 3))
        for context in range(X.shape[2]):
            intensity = X[:, :, context] / np.max(X[:, :, context])
            for rgb in range(3):
                data[:, :, rgb] = np.nansum(
                    np.concatenate([data[:, :, rgb][..., None], ((1-rgb_colors[context][rgb]) * intensity)[..., None]], axis=2), 
                    axis=2
                )
        if y_lim[0] < y_lim[1]:
            data = data[::-1]
        data = 1 - data
        
        ax.imshow(data)
        n_pixels = X.shape[0]
        y_ticks_pixels = self.map_to_pixel_space(n_pixels, y_lim, y_ticks)
        y_ticks_pixels = np.sort(y_ticks_pixels)
        y_ticks = np.sort(y_ticks)[::-1]
        
        ax.set_yticks(y_ticks_pixels)
        ax.set_yticklabels(y_ticks)
    
    def map_to_pixel_space(self, n_pixels: int, lims: List[float], y_ticks: List[float]):
        lims = np.sort(lims)[::-1]
        y_ticks_pixels = (n_pixels - 1) * ((np.array(y_ticks)-lims[0]) / (lims[1] - lims[0])) + 1 # TODO: do we need +1 in python?
        
        return y_ticks_pixels
    
    def colors(self):
        C = {}
        C["contexts"] = np.array([
            [0.1216, 0.4706, 0.7059, ], 
            [0.8902, 0.1020, 0.1098], 
            [1.0000, 0.4980, 0], 
            [0.2000, 0.6275, 0.1725], 
            [0.9843, 0.6039, 0.6000], 
            [0.8902, 0.1020, 0.1098], 
            [0.9922, 0.7490, 0.4353], 
            [1.0000, 0.4980, 0], 
        ])
        C["new_context"] = [[0.7, 0.7, 0.7]]
        C["marginal"] = [[208/255, 149/255, 213/255]]
        C["mean_of_marginal"] = [54/255, 204/255, 255/255]
        if (self.cues is not None) and len(self.cues) > 0:
            C["cues"] = get_cmap("copper", np.max(self.cues))

        return C
    