"""
Dynamic Programming Routines for the Rust (1987) Optimal Replacement Model

This module implements the core dynamic programming algorithms for solving
the optimal replacement problem, including value function iteration and
policy computation.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from numba import jit
import time


class DynamicProgram:
    """
    Dynamic Programming solver for the optimal replacement problem.
    
    Solves the Bellman equation:
    V(x) = max { u₀(x) + β ∫ V(y) dF(y|x),  u₁(0) + β ∫ V(y) dF(y|0) }
    
    where:
    - V(x) is the value function at state x
    - u₀(x) is the utility of keeping the asset (negative maintenance cost)
    - u₁(0) is the utility of replacement (negative replacement cost)
    - β is the discount factor
    - F(y|x) is the state transition probability
    """
    
    def __init__(self,
                 n_states: int,
                 discount_factor: float,
                 transition_matrix: np.ndarray,
                 maintenance_cost: np.ndarray,
                 replacement_cost: float):
        """
        Initialize Dynamic Programming solver.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        discount_factor : float
            Discount factor β ∈ (0, 1)
        transition_matrix : np.ndarray
            State transition probability matrix P[i,j] = Pr(x'=j|x=i)
        maintenance_cost : np.ndarray
            Cost of maintaining asset at each state
        replacement_cost : float
            Fixed cost of replacing the asset
        """
        self.n_states = n_states
        self.beta = discount_factor
        self.P = transition_matrix
        self.c_maintain = maintenance_cost
        self.c_replace = replacement_cost
        
        # Initialize value function and policy
        self.value_function = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)
        
        # Convergence tracking
        self.iteration_count = 0
        self.converged = False
    
    def bellman_operator(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the Bellman operator to value function V.
        
        Parameters
        ----------
        V : np.ndarray
            Current value function
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (new_value_function, optimal_policy)
        """
        # Expected value of keeping the asset at each state
        EV_keep = self.P @ V
        V_keep = -self.c_maintain + self.beta * EV_keep
        
        # Expected value of replacing the asset (start at state 0)
        EV_replace = self.P[0, :] @ V
        V_replace = -self.c_replace + self.beta * EV_replace
        
        # Choose optimal action: 0 = keep, 1 = replace
        V_new = np.maximum(V_keep, V_replace)
        policy = (V_replace > V_keep).astype(int)
        
        return V_new, policy
    
    def value_iteration(self,
                       tol: float = 1e-6,
                       max_iter: int = 1000,
                       verbose: bool = False) -> Dict:
        """
        Solve for optimal value function using value iteration.
        
        Parameters
        ----------
        tol : float
            Convergence tolerance (sup norm)
        max_iter : int
            Maximum number of iterations
        verbose : bool
            Print iteration progress
        
        Returns
        -------
        Dict
            Results dictionary with value function, policy, and convergence info
        """
        start_time = time.time()
        V = np.zeros(self.n_states)
        
        for iteration in range(max_iter):
            V_new, policy = self.bellman_operator(V)
            
            # Check convergence
            diff = np.max(np.abs(V_new - V))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: max diff = {diff:.8f}")
            
            if diff < tol:
                self.converged = True
                self.iteration_count = iteration + 1
                self.value_function = V_new
                self.policy = policy
                
                if verbose:
                    print(f"Converged in {iteration + 1} iterations")
                
                break
            
            V = V_new
        
        if not self.converged:
            print(f"Warning: Did not converge in {max_iter} iterations")
            self.value_function = V_new
            self.policy = policy
            self.iteration_count = max_iter
        
        elapsed_time = time.time() - start_time
        
        return {
            'value_function': self.value_function,
            'policy': self.policy,
            'converged': self.converged,
            'iterations': self.iteration_count,
            'time': elapsed_time,
            'final_diff': diff
        }
    
    def policy_iteration(self,
                        tol: float = 1e-6,
                        max_iter: int = 100,
                        verbose: bool = False) -> Dict:
        """
        Solve for optimal policy using policy iteration.
        
        Policy iteration alternates between:
        1. Policy evaluation: solve for V given policy π
        2. Policy improvement: update π to be greedy w.r.t. V
        
        Parameters
        ----------
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum number of iterations
        verbose : bool
            Print iteration progress
        
        Returns
        -------
        Dict
            Results dictionary with value function, policy, and convergence info
        """
        start_time = time.time()
        
        # Initialize with arbitrary policy (all keep)
        policy = np.zeros(self.n_states, dtype=int)
        
        for iteration in range(max_iter):
            # Policy evaluation: solve linear system
            V = self._policy_evaluation(policy)
            
            # Policy improvement
            _, policy_new = self.bellman_operator(V)
            
            # Check if policy converged
            if np.array_equal(policy_new, policy):
                self.converged = True
                self.iteration_count = iteration + 1
                self.value_function = V
                self.policy = policy_new
                
                if verbose:
                    print(f"Policy converged in {iteration + 1} iterations")
                
                break
            
            policy = policy_new
        
        if not self.converged:
            print(f"Warning: Policy did not converge in {max_iter} iterations")
            self.value_function = V
            self.policy = policy
            self.iteration_count = max_iter
        
        elapsed_time = time.time() - start_time
        
        return {
            'value_function': self.value_function,
            'policy': self.policy,
            'converged': self.converged,
            'iterations': self.iteration_count,
            'time': elapsed_time
        }
    
    def _policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        """
        Evaluate value function for a given policy.
        
        Solves: V = u + β P_π V
        where P_π is the transition matrix under policy π
        
        Parameters
        ----------
        policy : np.ndarray
            Policy array (0 = keep, 1 = replace)
        
        Returns
        -------
        np.ndarray
            Value function under the policy
        """
        # Construct flow utility under policy
        u = np.where(policy == 0, -self.c_maintain, -self.c_replace)
        
        # Construct transition matrix under policy
        P_policy = np.copy(self.P)
        replace_states = np.where(policy == 1)[0]
        for state in replace_states:
            P_policy[state, :] = self.P[0, :]
        
        # Solve linear system: (I - β P_π) V = u
        I = np.eye(self.n_states)
        V = np.linalg.solve(I - self.beta * P_policy, u)
        
        return V
    
    def get_choice_probabilities(self, 
                                V: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get choice probabilities at each state (for models with random utility).
        
        This is used in the NFXP estimation with Type I Extreme Value errors.
        
        Parameters
        ----------
        V : np.ndarray, optional
            Value function (uses self.value_function if None)
        
        Returns
        -------
        np.ndarray
            Probability of replacement at each state
        """
        if V is None:
            V = self.value_function
        
        # Expected value of each choice
        EV_keep = self.P @ V
        V_keep = -self.c_maintain + self.beta * EV_keep
        
        EV_replace = self.P[0, :] @ V
        V_replace = -self.c_replace + self.beta * EV_replace
        
        # In basic model, policy is deterministic
        # This returns "probability" as 0 or 1
        prob_replace = (V_replace > V_keep).astype(float)
        
        return prob_replace


@jit(nopython=True)
def fast_value_iteration(n_states: int,
                        beta: float,
                        P: np.ndarray,
                        c_maintain: np.ndarray,
                        c_replace: float,
                        tol: float = 1e-6,
                        max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fast value iteration using Numba JIT compilation.
    
    Parameters
    ----------
    n_states : int
        Number of states
    beta : float
        Discount factor
    P : np.ndarray
        Transition matrix
    c_maintain : np.ndarray
        Maintenance costs
    c_replace : float
        Replacement cost
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (value_function, policy, iterations)
    """
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=np.int32)
    
    for iteration in range(max_iter):
        V_old = V.copy()
        
        for i in range(n_states):
            # Value of keeping
            EV_keep = 0.0
            for j in range(n_states):
                EV_keep += P[i, j] * V_old[j]
            V_keep = -c_maintain[i] + beta * EV_keep
            
            # Value of replacing
            EV_replace = 0.0
            for j in range(n_states):
                EV_replace += P[0, j] * V_old[j]
            V_replace = -c_replace + beta * EV_replace
            
            # Choose best action
            if V_replace > V_keep:
                V[i] = V_replace
                policy[i] = 1
            else:
                V[i] = V_keep
                policy[i] = 0
        
        # Check convergence
        diff = np.max(np.abs(V - V_old))
        if diff < tol:
            return V, policy, iteration + 1
    
    return V, policy, max_iter


def solve_dynamic_program(n_states: int,
                         discount_factor: float,
                         transition_matrix: np.ndarray,
                         maintenance_cost: np.ndarray,
                         replacement_cost: float,
                         method: str = "value_iteration",
                         use_numba: bool = True,
                         **kwargs) -> Dict:
    """
    Convenience function to solve the dynamic program.
    
    Parameters
    ----------
    n_states : int
        Number of discrete states
    discount_factor : float
        Discount factor
    transition_matrix : np.ndarray
        Transition probability matrix
    maintenance_cost : np.ndarray
        Maintenance cost function
    replacement_cost : float
        Replacement cost
    method : str
        Solution method ("value_iteration" or "policy_iteration")
    use_numba : bool
        Use Numba-accelerated solver
    **kwargs
        Additional arguments for solver
    
    Returns
    -------
    Dict
        Solution dictionary
    
    Examples
    --------
    >>> P = np.eye(90) * 0.8 + np.eye(90, k=1) * 0.2
    >>> c_maintain = np.linspace(0, 10, 90)
    >>> result = solve_dynamic_program(90, 0.95, P, c_maintain, 10.0)
    """
    if use_numba and method == "value_iteration":
        V, policy, iterations = fast_value_iteration(
            n_states, discount_factor, transition_matrix,
            maintenance_cost, replacement_cost,
            tol=kwargs.get('tol', 1e-6),
            max_iter=kwargs.get('max_iter', 1000)
        )
        return {
            'value_function': V,
            'policy': policy,
            'iterations': iterations,
            'converged': True
        }
    else:
        dp = DynamicProgram(n_states, discount_factor, transition_matrix,
                          maintenance_cost, replacement_cost)
        
        if method == "value_iteration":
            return dp.value_iteration(**kwargs)
        elif method == "policy_iteration":
            return dp.policy_iteration(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
