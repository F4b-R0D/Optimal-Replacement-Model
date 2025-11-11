"""
Simulation Utilities for the Rust (1987) Optimal Replacement Model

This module provides tools for simulating optimal replacement behavior
and generating synthetic datasets for testing and validation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
import pandas as pd


class ReplacementSimulator:
    """
    Simulator for optimal replacement decisions.
    
    Simulates agent behavior under optimal or estimated policies,
    generating synthetic panel data of states and decisions.
    """
    
    def __init__(self,
                 n_states: int,
                 transition_matrix: np.ndarray,
                 policy: Optional[np.ndarray] = None,
                 choice_probs: Optional[np.ndarray] = None,
                 random_state: Optional[int] = None):
        """
        Initialize simulator.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        transition_matrix : np.ndarray
            State transition probability matrix
        policy : np.ndarray, optional
            Deterministic policy (0=keep, 1=replace)
        choice_probs : np.ndarray, optional
            Stochastic choice probabilities for replacement
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_states = n_states
        self.P = transition_matrix
        self.policy = policy
        self.choice_probs = choice_probs
        self.rng = np.random.RandomState(random_state)
        
        if policy is None and choice_probs is None:
            raise ValueError("Must provide either policy or choice_probs")
    
    def simulate_trajectory(self,
                           n_periods: int,
                           initial_state: int = 0,
                           return_dataframe: bool = True) -> pd.DataFrame or Tuple:
        """
        Simulate a single agent's trajectory.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        initial_state : int
            Starting state
        return_dataframe : bool
            Return results as DataFrame vs arrays
        
        Returns
        -------
        pd.DataFrame or Tuple
            Simulated states and decisions
        """
        states = np.zeros(n_periods, dtype=int)
        decisions = np.zeros(n_periods, dtype=int)
        
        current_state = initial_state
        
        for t in range(n_periods):
            states[t] = current_state
            
            # Make decision
            if self.choice_probs is not None:
                # Stochastic choice based on probabilities
                prob_replace = self.choice_probs[current_state]
                decision = self.rng.binomial(1, prob_replace)
            else:
                # Deterministic choice from policy
                decision = self.policy[current_state]
            
            decisions[t] = decision
            
            # Transition to next state
            if decision == 1:
                # Replacement: restart at state 0
                next_state = 0
            else:
                # Keep: transition according to P
                next_state = self.rng.choice(
                    self.n_states,
                    p=self.P[current_state, :]
                )
            
            current_state = next_state
        
        if return_dataframe:
            df = pd.DataFrame({
                'period': np.arange(n_periods),
                'state': states,
                'decision': decisions,
                'agent': 0
            })
            return df
        else:
            return states, decisions
    
    def simulate_panel(self,
                      n_agents: int,
                      n_periods: int,
                      initial_state_dist: Optional[np.ndarray] = None,
                      return_dataframe: bool = True) -> pd.DataFrame or Dict:
        """
        Simulate panel data with multiple agents.
        
        Parameters
        ----------
        n_agents : int
            Number of agents to simulate
        n_periods : int
            Number of periods per agent
        initial_state_dist : np.ndarray, optional
            Distribution over initial states (uniform if None)
        return_dataframe : bool
            Return as DataFrame vs dictionary of arrays
        
        Returns
        -------
        pd.DataFrame or Dict
            Panel dataset with states and decisions
        """
        if initial_state_dist is None:
            initial_state_dist = np.ones(self.n_states) / self.n_states
        
        all_data = []
        
        for agent in range(n_agents):
            # Draw initial state
            initial_state = self.rng.choice(
                self.n_states,
                p=initial_state_dist
            )
            
            # Simulate trajectory
            df = self.simulate_trajectory(
                n_periods,
                initial_state=initial_state,
                return_dataframe=True
            )
            df['agent'] = agent
            all_data.append(df)
        
        if return_dataframe:
            return pd.concat(all_data, ignore_index=True)
        else:
            combined_df = pd.concat(all_data, ignore_index=True)
            return {
                'states': combined_df['state'].values,
                'decisions': combined_df['decision'].values,
                'agents': combined_df['agent'].values,
                'periods': combined_df['period'].values
            }
    
    def compute_statistics(self, states: np.ndarray, decisions: np.ndarray) -> Dict:
        """
        Compute summary statistics from simulated data.
        
        Parameters
        ----------
        states : np.ndarray
            Simulated states
        decisions : np.ndarray
            Simulated decisions
        
        Returns
        -------
        Dict
            Summary statistics
        """
        stats = {
            'mean_state': np.mean(states),
            'median_state': np.median(states),
            'std_state': np.std(states),
            'replacement_rate': np.mean(decisions),
            'n_observations': len(states),
            'state_distribution': np.bincount(states, minlength=self.n_states) / len(states)
        }
        return stats


def generate_synthetic_data(n_agents: int,
                           n_periods: int,
                           n_states: int,
                           transition_probs: np.ndarray,
                           maintenance_cost_params: np.ndarray,
                           replacement_cost: float,
                           discount_factor: float,
                           noise_level: float = 0.0,
                           random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic dataset from known parameters.
    
    This is useful for testing estimation procedures and validating
    the implementation.
    
    Parameters
    ----------
    n_agents : int
        Number of agents
    n_periods : int
        Number of periods per agent
    n_states : int
        Number of discrete states
    transition_probs : np.ndarray
        Transition probability vector for increments
    maintenance_cost_params : np.ndarray
        Parameters for linear cost: [θ₀, θ₁]
    replacement_cost : float
        Replacement cost
    discount_factor : float
        Discount factor
    noise_level : float
        Standard deviation of additive noise to choice probabilities
    random_state : int, optional
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Synthetic panel dataset
    """
    from .transitions import DiscreteTransition
    from .cost_functions import LinearCost
    from .dp import DynamicProgram
    
    # Create transition model
    trans_model = DiscreteTransition(n_states, transition_probs)
    P = trans_model.get_transition_matrix()
    
    # Create cost function
    cost_fn = LinearCost(maintenance_cost_params)
    state_grid = np.arange(n_states)
    maintenance_costs = cost_fn.evaluate(state_grid)
    
    # Solve DP for optimal policy
    dp = DynamicProgram(n_states, discount_factor, P,
                       maintenance_costs, replacement_cost)
    result = dp.value_iteration(verbose=False)
    
    # Get choice probabilities
    choice_probs = dp.get_choice_probabilities()
    
    # Add noise if specified
    if noise_level > 0:
        rng = np.random.RandomState(random_state)
        noise = rng.normal(0, noise_level, size=n_states)
        choice_probs = np.clip(choice_probs + noise, 0, 1)
    
    # Simulate data
    simulator = ReplacementSimulator(
        n_states, P,
        choice_probs=choice_probs,
        random_state=random_state
    )
    
    df = simulator.simulate_panel(n_agents, n_periods)
    
    # Add metadata
    df.attrs['true_parameters'] = {
        'maintenance_cost_params': maintenance_cost_params,
        'replacement_cost': replacement_cost,
        'discount_factor': discount_factor,
        'transition_probs': transition_probs
    }
    
    return df


def analyze_replacement_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze replacement patterns in simulated or real data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data with columns: agent, period, state, decision
    
    Returns
    -------
    Dict
        Analysis results including hazard rates, survival curves, etc.
    """
    results = {}
    
    # Overall statistics
    results['total_observations'] = len(df)
    results['n_agents'] = df['agent'].nunique()
    results['n_periods'] = df.groupby('agent')['period'].count().mean()
    results['replacement_rate'] = df['decision'].mean()
    
    # State distribution
    results['state_distribution'] = df['state'].value_counts(normalize=True).sort_index()
    
    # Hazard rate (replacement probability by state)
    hazard = df.groupby('state')['decision'].mean()
    results['hazard_rate'] = hazard
    
    # Survival function (probability of not replacing up to state x)
    survival = 1 - hazard.cumsum() / len(hazard)
    results['survival_function'] = survival
    
    # Time between replacements
    if 'agent' in df.columns:
        time_between = []
        for agent in df['agent'].unique():
            agent_data = df[df['agent'] == agent]
            replacement_periods = agent_data[agent_data['decision'] == 1]['period'].values
            if len(replacement_periods) > 1:
                time_between.extend(np.diff(replacement_periods))
        
        if time_between:
            results['mean_time_between_replacements'] = np.mean(time_between)
            results['median_time_between_replacements'] = np.median(time_between)
    
    return results


def create_rust_dataset(sample_size: int = 1000,
                       n_states: int = 90,
                       random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Create a dataset mimicking Rust's (1987) bus engine replacement data.
    
    Parameters
    ----------
    sample_size : int
        Number of observations
    n_states : int
        Number of mileage bins (typically ~90 for Rust data)
    random_state : int, optional
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Synthetic Rust-style dataset
    """
    # Parameters approximately matching Rust (1987) estimates
    maintenance_cost_params = np.array([0.0, 0.001])  # Linear in thousands
    replacement_cost = 11.7  # Thousands of dollars
    discount_factor = 0.9999  # Monthly discount
    transition_probs = np.array([0.349, 0.639, 0.012])  # Mileage increments
    
    # Number of buses and periods
    n_buses = max(10, sample_size // 100)
    n_periods = sample_size // n_buses
    
    df = generate_synthetic_data(
        n_agents=n_buses,
        n_periods=n_periods,
        n_states=n_states,
        transition_probs=transition_probs,
        maintenance_cost_params=maintenance_cost_params,
        replacement_cost=replacement_cost,
        discount_factor=discount_factor,
        random_state=random_state
    )
    
    # Rename for clarity
    df.rename(columns={'agent': 'bus_id', 'state': 'mileage_bin'}, inplace=True)
    
    return df
