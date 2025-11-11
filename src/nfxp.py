"""
NFXP (Nested Fixed Point) Estimation for the Rust (1987) Model

This module implements the nested fixed point algorithm for maximum likelihood
estimation of structural parameters in dynamic discrete choice models.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from typing import Dict, Tuple, Optional, Callable
import time
from .dp import DynamicProgram, fast_value_iteration


class NFXPEstimator:
    """
    Nested Fixed Point Maximum Likelihood Estimator.
    
    The NFXP algorithm (Rust, 1987) estimates structural parameters by:
    1. Outer loop: Search over parameters using optimization
    2. Inner loop: For each parameter vector, solve the DP (fixed point)
    3. Evaluate likelihood of observed data given solution
    
    The likelihood incorporates Type I Extreme Value errors:
    P(d=1|x,θ) = exp(V₁(x,θ)) / [exp(V₀(x,θ)) + exp(V₁(x,θ))]
    """
    
    def __init__(self,
                 n_states: int,
                 discount_factor: float,
                 transition_matrix: np.ndarray):
        """
        Initialize NFXP estimator.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        discount_factor : float
            Discount factor (typically fixed)
        transition_matrix : np.ndarray
            State transition probability matrix
        """
        self.n_states = n_states
        self.beta = discount_factor
        self.P = transition_matrix
        
        # Storage for estimation results
        self.theta_hat = None
        self.standard_errors = None
        self.log_likelihood = None
        self.convergence_info = {}
    
    def solve_inner_problem(self,
                           maintenance_cost: np.ndarray,
                           replacement_cost: float,
                           tol: float = 1e-6) -> np.ndarray:
        """
        Solve the inner fixed point problem (DP solution).
        
        Parameters
        ----------
        maintenance_cost : np.ndarray
            Maintenance cost at each state
        replacement_cost : float
            Replacement cost
        tol : float
            Convergence tolerance for value iteration
        
        Returns
        -------
        np.ndarray
            Value function at each state
        """
        V, _, _ = fast_value_iteration(
            self.n_states, self.beta, self.P,
            maintenance_cost, replacement_cost, tol=tol
        )
        return V
    
    def choice_probability(self,
                          V: np.ndarray,
                          maintenance_cost: np.ndarray,
                          replacement_cost: float) -> np.ndarray:
        """
        Compute choice probabilities with Type I Extreme Value errors.
        
        P(replace | x, θ) = exp(V_replace) / [exp(V_keep) + exp(V_replace)]
        
        Parameters
        ----------
        V : np.ndarray
            Value function
        maintenance_cost : np.ndarray
            Maintenance cost function
        replacement_cost : float
            Replacement cost
        
        Returns
        -------
        np.ndarray
            Probability of replacement at each state
        """
        # Expected continuation value
        EV = self.P @ V
        
        # Value of each choice (flow utility + continuation)
        V_keep = -maintenance_cost + self.beta * EV
        V_replace = -replacement_cost + self.beta * self.P[0, :] @ V
        
        # Logit choice probability (using log-sum-exp for numerical stability)
        max_V = np.maximum(V_keep, V_replace)
        log_sum = max_V + np.log(
            np.exp(V_keep - max_V) + np.exp(V_replace - max_V)
        )
        prob_replace = np.exp(V_replace - log_sum)
        
        return prob_replace
    
    def log_likelihood_function(self,
                                theta: np.ndarray,
                                states: np.ndarray,
                                decisions: np.ndarray,
                                cost_function: Callable,
                                return_components: bool = False) -> float:
        """
        Compute log-likelihood for given parameters and data.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [cost_params..., replacement_cost]
        states : np.ndarray
            Observed states for each observation
        decisions : np.ndarray
            Observed decisions (0=keep, 1=replace)
        cost_function : Callable
            Function that maps (states, cost_params) -> costs
        return_components : bool
            If True, return individual log-likelihood contributions
        
        Returns
        -------
        float or np.ndarray
            Log-likelihood value(s)
        """
        # Parse parameters
        cost_params = theta[:-1]
        replacement_cost = theta[-1]
        
        # Compute maintenance costs
        state_grid = np.arange(self.n_states)
        maintenance_cost = cost_function(state_grid, cost_params)
        
        # Solve inner problem
        V = self.solve_inner_problem(maintenance_cost, replacement_cost)
        
        # Get choice probabilities
        prob_replace = self.choice_probability(V, maintenance_cost, replacement_cost)
        
        # Compute log-likelihood contributions
        # Clip probabilities to avoid log(0)
        prob_replace = np.clip(prob_replace, 1e-10, 1 - 1e-10)
        prob_keep = 1 - prob_replace
        
        # Map states to probabilities
        probs_at_states = np.where(
            decisions == 1,
            prob_replace[states],
            prob_keep[states]
        )
        
        log_likes = np.log(probs_at_states)
        
        if return_components:
            return log_likes
        else:
            return np.sum(log_likes)
    
    def estimate(self,
                states: np.ndarray,
                decisions: np.ndarray,
                cost_function: Callable,
                theta_init: Optional[np.ndarray] = None,
                method: str = 'BFGS',
                verbose: bool = True) -> Dict:
        """
        Estimate structural parameters using NFXP.
        
        Parameters
        ----------
        states : np.ndarray
            Observed states (integer indices)
        decisions : np.ndarray
            Observed decisions (0=keep, 1=replace)
        cost_function : Callable
            Cost function: (states, params) -> costs
        theta_init : np.ndarray, optional
            Initial parameter guess
        method : str
            Optimization method for scipy.optimize.minimize
        verbose : bool
            Print estimation progress
        
        Returns
        -------
        Dict
            Estimation results including parameters, standard errors, etc.
        """
        start_time = time.time()
        
        # Initialize parameters if not provided
        if theta_init is None:
            # Default: linear cost with small coefficients + replacement cost
            theta_init = np.array([0.1, 0.05, 8.0])
        
        if verbose:
            print("Starting NFXP estimation...")
            print(f"Initial parameters: {theta_init}")
            print(f"Sample size: {len(states)}")
        
        # Define objective (negative log-likelihood)
        iteration_count = [0]
        
        def objective(theta):
            iteration_count[0] += 1
            ll = self.log_likelihood_function(theta, states, decisions, cost_function)
            
            if verbose and iteration_count[0] % 10 == 0:
                print(f"Iteration {iteration_count[0]}: LL = {ll:.4f}, theta = {theta}")
            
            return -ll  # Minimize negative LL
        
        # Optimize
        result = minimize(
            objective,
            theta_init,
            method=method,
            options={'maxiter': 500, 'disp': verbose}
        )
        
        elapsed_time = time.time() - start_time
        
        # Store results
        self.theta_hat = result.x
        self.log_likelihood = -result.fun
        self.convergence_info = {
            'success': result.success,
            'message': result.message,
            'iterations': iteration_count[0],
            'time': elapsed_time
        }
        
        # Compute standard errors (inverse Hessian approximation)
        if result.hess_inv is not None:
            if isinstance(result.hess_inv, np.ndarray):
                self.standard_errors = np.sqrt(np.diag(result.hess_inv))
            else:
                # For BFGS, hess_inv is a LbfgsInvHessProduct object
                try:
                    hess_inv = result.hess_inv.todense()
                    self.standard_errors = np.sqrt(np.diag(hess_inv))
                except:
                    self.standard_errors = None
        
        if verbose:
            print("\nEstimation complete!")
            print(f"Parameters: {self.theta_hat}")
            print(f"Log-likelihood: {self.log_likelihood:.4f}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
        
        return {
            'theta': self.theta_hat,
            'log_likelihood': self.log_likelihood,
            'standard_errors': self.standard_errors,
            'convergence': self.convergence_info,
            'sample_size': len(states)
        }
    
    def predict(self,
               theta: Optional[np.ndarray] = None,
               cost_function: Optional[Callable] = None) -> Dict:
        """
        Predict optimal policy and value function for given parameters.
        
        Parameters
        ----------
        theta : np.ndarray, optional
            Parameter vector (uses estimated parameters if None)
        cost_function : Callable, optional
            Cost function
        
        Returns
        -------
        Dict
            Predictions including value function, policy, choice probabilities
        """
        if theta is None:
            theta = self.theta_hat
        
        if theta is None:
            raise ValueError("No parameters available. Run estimate() first.")
        
        # Parse parameters
        cost_params = theta[:-1]
        replacement_cost = theta[-1]
        
        # Compute costs
        state_grid = np.arange(self.n_states)
        maintenance_cost = cost_function(state_grid, cost_params)
        
        # Solve DP
        V = self.solve_inner_problem(maintenance_cost, replacement_cost)
        prob_replace = self.choice_probability(V, maintenance_cost, replacement_cost)
        
        # Deterministic policy
        policy = (prob_replace > 0.5).astype(int)
        
        return {
            'value_function': V,
            'policy': policy,
            'choice_probabilities': prob_replace,
            'maintenance_cost': maintenance_cost,
            'replacement_cost': replacement_cost
        }


def linear_cost_wrapper(states: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear cost function wrapper for NFXP estimation.
    
    c(x) = θ₀ + θ₁ * x
    
    Parameters
    ----------
    states : np.ndarray
        State values
    params : np.ndarray
        Cost parameters [θ₀, θ₁]
    
    Returns
    -------
    np.ndarray
        Costs at each state
    """
    return params[0] + params[1] * states


def quadratic_cost_wrapper(states: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic cost function wrapper for NFXP estimation.
    
    c(x) = θ₀ + θ₁ * x + θ₂ * x²
    
    Parameters
    ----------
    states : np.ndarray
        State values
    params : np.ndarray
        Cost parameters [θ₀, θ₁, θ₂]
    
    Returns
    -------
    np.ndarray
        Costs at each state
    """
    return params[0] + params[1] * states + params[2] * states**2
