"""
State Transition Probabilities for the Rust (1987) Model

This module implements state transition probability functions for modeling
how assets age/deteriorate over time.
"""

import numpy as np
from scipy.stats import poisson
from typing import Optional, Tuple
from numba import jit


class TransitionProbability:
    """
    Base class for state transition probabilities.
    
    In the Rust (1987) model, transitions represent how mileage accumulates
    between periods following a discrete probability distribution.
    """
    
    def __init__(self, n_states: int):
        """
        Initialize transition probability object.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        """
        self.n_states = n_states
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the full transition probability matrix.
        
        Returns
        -------
        np.ndarray
            Transition matrix P where P[i,j] = Pr(x_{t+1}=j | x_t=i)
        """
        raise NotImplementedError("Subclasses must implement get_transition_matrix")


class DiscreteTransition(TransitionProbability):
    """
    Discrete transition probabilities with finite support.
    
    Models mileage increments as discrete jumps (e.g., 0, 1, 2, ... miles).
    This is the approach used in Rust (1987).
    """
    
    def __init__(self, n_states: int, transition_probs: Optional[np.ndarray] = None):
        """
        Initialize discrete transition probability.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        transition_probs : np.ndarray, optional
            Probability vector for state increments [p(Δx=0), p(Δx=1), ...]
            If None, uses a default geometric distribution
        """
        super().__init__(n_states)
        
        if transition_probs is None:
            # Default: geometric-like distribution for mileage increments
            # Most common increments are 0, 1, 2 thousand miles per month
            transition_probs = np.array([0.4, 0.35, 0.25])
        
        self.transition_probs = transition_probs / transition_probs.sum()
        self.max_increment = len(transition_probs) - 1
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Construct the full transition matrix.
        
        Returns
        -------
        np.ndarray
            n_states × n_states transition matrix
        """
        P = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(len(self.transition_probs)):
                next_state = min(i + j, self.n_states - 1)
                P[i, next_state] += self.transition_probs[j]
        
        return P
    
    def simulate_transition(self, current_state: int, 
                          random_state: Optional[np.random.RandomState] = None) -> int:
        """
        Simulate a single state transition.
        
        Parameters
        ----------
        current_state : int
            Current state index
        random_state : np.random.RandomState, optional
            Random state for reproducibility
        
        Returns
        -------
        int
            Next state index
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        increment = random_state.choice(
            len(self.transition_probs), 
            p=self.transition_probs
        )
        
        return min(current_state + increment, self.n_states - 1)


class PoissonTransition(TransitionProbability):
    """
    Poisson-distributed state transitions.
    
    Models mileage increments as following a Poisson distribution,
    which is appropriate for count data like miles driven per period.
    """
    
    def __init__(self, n_states: int, lambda_param: float = 1.5):
        """
        Initialize Poisson transition probability.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        lambda_param : float
            Mean of the Poisson distribution (average increment per period)
        """
        super().__init__(n_states)
        self.lambda_param = lambda_param
        
        # Pre-compute transition probabilities for efficiency
        max_increment = min(20, n_states)  # Truncate at reasonable value
        self.transition_probs = poisson.pmf(np.arange(max_increment), lambda_param)
        self.transition_probs /= self.transition_probs.sum()  # Normalize
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Construct the full transition matrix with Poisson increments.
        
        Returns
        -------
        np.ndarray
            n_states × n_states transition matrix
        """
        P = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(len(self.transition_probs)):
                next_state = min(i + j, self.n_states - 1)
                P[i, next_state] += self.transition_probs[j]
        
        return P


@jit(nopython=True)
def _fast_transition_matrix(n_states: int, probs: np.ndarray) -> np.ndarray:
    """
    Fast computation of transition matrix using Numba JIT compilation.
    
    Parameters
    ----------
    n_states : int
        Number of states
    probs : np.ndarray
        Transition probability vector
    
    Returns
    -------
    np.ndarray
        Transition matrix
    """
    P = np.zeros((n_states, n_states))
    max_inc = len(probs)
    
    for i in range(n_states):
        for j in range(max_inc):
            next_state = min(i + j, n_states - 1)
            P[i, next_state] += probs[j]
    
    return P


class FastDiscreteTransition(TransitionProbability):
    """
    Optimized discrete transition using Numba for large state spaces.
    """
    
    def __init__(self, n_states: int, transition_probs: Optional[np.ndarray] = None):
        """
        Initialize fast discrete transition probability.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        transition_probs : np.ndarray, optional
            Probability vector for state increments
        """
        super().__init__(n_states)
        
        if transition_probs is None:
            transition_probs = np.array([0.4, 0.35, 0.25])
        
        self.transition_probs = transition_probs / transition_probs.sum()
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get transition matrix using optimized Numba implementation.
        
        Returns
        -------
        np.ndarray
            n_states × n_states transition matrix
        """
        return _fast_transition_matrix(self.n_states, self.transition_probs)


def create_transition_model(model_type: str = "discrete",
                           n_states: int = 90,
                           **kwargs) -> TransitionProbability:
    """
    Factory function to create transition probability models.
    
    Parameters
    ----------
    model_type : str
        Type of transition model ("discrete", "poisson", "fast")
    n_states : int
        Number of discrete states
    **kwargs
        Additional parameters for specific models
    
    Returns
    -------
    TransitionProbability
        Instantiated transition probability object
    
    Examples
    --------
    >>> trans = create_transition_model("discrete", n_states=90)
    >>> P = trans.get_transition_matrix()
    >>> print(P.shape)
    (90, 90)
    """
    models = {
        "discrete": DiscreteTransition,
        "poisson": PoissonTransition,
        "fast": FastDiscreteTransition
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](n_states, **kwargs)
