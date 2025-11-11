"""
Cost Functions for the Rust (1987) Optimal Replacement Model

This module implements various cost functions used in the dynamic programming
formulation of the optimal replacement problem.
"""

import numpy as np
from typing import Optional, Tuple


class CostFunction:
    """
    Base class for cost functions in the optimal replacement model.
    
    In the Rust (1987) model, the cost function represents the maintenance cost
    as a function of the asset's state (typically mileage or age).
    """
    
    def __init__(self, theta: Optional[np.ndarray] = None):
        """
        Initialize cost function with parameters.
        
        Parameters
        ----------
        theta : np.ndarray, optional
            Parameter vector for the cost function
        """
        self.theta = theta if theta is not None else np.array([0.0, 0.1])
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the cost function at state(s) x.
        
        Parameters
        ----------
        x : np.ndarray
            State variable(s) (e.g., mileage)
        
        Returns
        -------
        np.ndarray
            Cost at each state
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class LinearCost(CostFunction):
    """
    Linear cost function: c(x, θ) = θ₀ + θ₁ * x
    
    This is the simplest parametric form used in Rust (1987).
    """
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate linear cost function.
        
        Parameters
        ----------
        x : np.ndarray
            State variable (mileage)
        
        Returns
        -------
        np.ndarray
            Linear cost θ₀ + θ₁ * x
        """
        return self.theta[0] + self.theta[1] * x


class QuadraticCost(CostFunction):
    """
    Quadratic cost function: c(x, θ) = θ₀ + θ₁ * x + θ₂ * x²
    
    Allows for increasing marginal costs as the asset ages.
    """
    
    def __init__(self, theta: Optional[np.ndarray] = None):
        """Initialize quadratic cost function."""
        super().__init__(theta)
        if self.theta.shape[0] < 3:
            self.theta = np.array([0.0, 0.1, 0.01])
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate quadratic cost function.
        
        Parameters
        ----------
        x : np.ndarray
            State variable (mileage)
        
        Returns
        -------
        np.ndarray
            Quadratic cost θ₀ + θ₁ * x + θ₂ * x²
        """
        return self.theta[0] + self.theta[1] * x + self.theta[2] * x**2


class PolynomialCost(CostFunction):
    """
    Polynomial cost function for more flexible specifications.
    
    c(x, θ) = Σᵢ θᵢ * xⁱ for i = 0, 1, ..., degree
    """
    
    def __init__(self, theta: Optional[np.ndarray] = None, degree: int = 2):
        """
        Initialize polynomial cost function.
        
        Parameters
        ----------
        theta : np.ndarray, optional
            Parameter vector of length (degree + 1)
        degree : int
            Degree of the polynomial
        """
        self.degree = degree
        if theta is None:
            theta = np.zeros(degree + 1)
            theta[1] = 0.1
        super().__init__(theta)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial cost function.
        
        Parameters
        ----------
        x : np.ndarray
            State variable (mileage)
        
        Returns
        -------
        np.ndarray
            Polynomial cost Σᵢ θᵢ * xⁱ
        """
        cost = np.zeros_like(x, dtype=float)
        for i, coef in enumerate(self.theta):
            cost += coef * x**i
        return cost


class ReplacementCost:
    """
    Represents the cost of replacing an asset.
    
    In the Rust (1987) model, this is the parameter RC (Replacement Cost).
    """
    
    def __init__(self, rc: float = 10.0):
        """
        Initialize replacement cost.
        
        Parameters
        ----------
        rc : float
            Fixed cost of replacement (in thousands of dollars)
        """
        self.rc = rc
    
    def get_cost(self) -> float:
        """
        Get the replacement cost.
        
        Returns
        -------
        float
            Replacement cost
        """
        return self.rc


def create_cost_function(cost_type: str = "linear", 
                        theta: Optional[np.ndarray] = None) -> CostFunction:
    """
    Factory function to create cost functions.
    
    Parameters
    ----------
    cost_type : str
        Type of cost function ("linear", "quadratic", "polynomial")
    theta : np.ndarray, optional
        Parameter vector
    
    Returns
    -------
    CostFunction
        Instantiated cost function object
    
    Examples
    --------
    >>> cost_fn = create_cost_function("linear", theta=np.array([0.5, 0.1]))
    >>> x = np.array([0, 10, 20, 30])
    >>> costs = cost_fn.evaluate(x)
    """
    cost_types = {
        "linear": LinearCost,
        "quadratic": QuadraticCost,
        "polynomial": PolynomialCost
    }
    
    if cost_type not in cost_types:
        raise ValueError(f"Unknown cost type: {cost_type}. "
                        f"Available: {list(cost_types.keys())}")
    
    return cost_types[cost_type](theta)
