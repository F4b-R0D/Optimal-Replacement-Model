# Optimal Replacement Model

A comprehensive implementation of the Rust (1987) dynamic discrete choice model for optimal asset replacement, with applications to physical asset management and capital budgeting.

## Overview

This repository provides a complete framework for modeling optimal replacement decisions using dynamic programming and structural estimation. The model is based on Harold Zurcher's bus engine replacement problem from Rust (1987), which has become a foundational example in structural econometrics.

### Key Features

- **Dynamic Programming Routines**: Value iteration and policy iteration algorithms
- **Cost Functions**: Flexible parametric specifications (linear, quadratic, polynomial)
- **State Transitions**: Discrete and Poisson transition models with Numba optimization
- **NFXP Estimation**: Nested Fixed Point Maximum Likelihood estimation
- **Simulation Tools**: Generate synthetic datasets and analyze replacement patterns
- **Automation Pipelines**: Complete workflows from data generation to estimation
- **Jupyter Notebooks**: Interactive explanations and examples

## Model Description

### The Rust (1987) Model

The model considers an agent (e.g., bus company manager) who observes the state of an asset (e.g., bus mileage) and must decide whether to:
1. **Keep** the asset and pay maintenance cost `c(x, θ)`
2. **Replace** the asset at cost `RC` and restart at state 0

The agent maximizes expected discounted utility:

```
V(x) = max { u₀(x) + β E[V(x')|x],  u₁(0) + β E[V(0')|0] }
```

Where:
- `V(x)` = value function at state x
- `u₀(x) = -c(x, θ)` = utility of keeping (negative maintenance cost)
- `u₁(0) = -RC` = utility of replacement (negative replacement cost)
- `β` = discount factor
- `x'` = next period state

### Bellman Equation

The value function satisfies the Bellman equation:

```
V(x) = max { -c(x,θ) + β ∫ V(y) dF(y|x),  -RC + β ∫ V(y) dF(y|0) }
```

where `F(y|x)` is the transition probability from state x to state y.

### Cost Function

The maintenance cost typically takes a linear or quadratic form:

**Linear**: `c(x, θ) = θ₀ + θ₁ × x`

**Quadratic**: `c(x, θ) = θ₀ + θ₁ × x + θ₂ × x²`

where x represents mileage (in thousands of miles) or age.

### Transition Probabilities

The state evolves according to a discrete probability distribution:

```
x_{t+1} = x_t + ε_t
```

where `ε_t ~ F(·)` represents the mileage increment (e.g., 0, 1, or 2 thousand miles per month).

### NFXP Estimation

The Nested Fixed Point (NFXP) algorithm estimates structural parameters `θ = [θ_cost, RC]` by:

1. **Outer Loop**: Search over parameter space using optimization
2. **Inner Loop**: For each parameter vector, solve the DP to get value function
3. **Likelihood**: Compute likelihood of observed choices given solution

With Type I Extreme Value errors, the replacement probability is:

```
P(replace | x, θ) = exp(V_replace) / [exp(V_keep) + exp(V_replace)]
```

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Numba >= 0.54.0
- Matplotlib >= 3.4.0
- Jupyter >= 1.0.0

### Setup

```bash
# Clone the repository
git clone https://github.com/F4b-R0D/Optimal-Replacement-Model.git
cd Optimal-Replacement-Model

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
Optimal-Replacement-Model/
├── data/                    # Sample datasets
├── src/                     # Core implementation
│   ├── __init__.py
│   ├── cost_functions.py   # Maintenance cost specifications
│   ├── transitions.py      # State transition models
│   ├── dp.py              # Dynamic programming algorithms
│   ├── nfxp.py            # NFXP estimation
│   └── simulation.py      # Data generation and simulation
├── automation/             # Automation scripts
│   ├── generate_data.py   # Generate sample datasets
│   └── run_pipeline.py    # Complete analysis workflow
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### 1. Generate Sample Data

```python
from src.simulation import create_rust_dataset

# Generate synthetic Rust (1987) bus replacement data
df = create_rust_dataset(sample_size=5000, n_states=90, random_state=42)
print(df.head())
```

### 2. Solve Dynamic Program

```python
import numpy as np
from src.transitions import create_transition_model
from src.cost_functions import create_cost_function
from src.dp import solve_dynamic_program

# Setup
n_states = 90
discount_factor = 0.9999

# Create transition matrix
trans_model = create_transition_model("discrete", n_states=n_states)
P = trans_model.get_transition_matrix()

# Define maintenance cost
cost_fn = create_cost_function("linear", theta=np.array([0.0, 0.001]))
maintenance_costs = cost_fn.evaluate(np.arange(n_states))

# Solve
result = solve_dynamic_program(
    n_states, discount_factor, P,
    maintenance_costs, replacement_cost=11.7,
    method="value_iteration"
)

print(f"Converged in {result['iterations']} iterations")
print(f"Optimal policy: {result['policy'][:20]}")  # First 20 states
```

### 3. Estimate Parameters

```python
from src.nfxp import NFXPEstimator, linear_cost_wrapper

# Initialize estimator
estimator = NFXPEstimator(n_states, discount_factor, P)

# Prepare data
states = df['mileage_bin'].values
decisions = df['decision'].values

# Estimate
theta_init = np.array([0.0, 0.002, 10.0])  # [θ₀, θ₁, RC]
results = estimator.estimate(
    states, decisions, linear_cost_wrapper,
    theta_init=theta_init, verbose=True
)

print(f"Estimated parameters: {results['theta']}")
print(f"Log-likelihood: {results['log_likelihood']:.2f}")
```

### 4. Run Complete Pipeline

```bash
cd automation
python run_pipeline.py
```

This will:
- Generate synthetic data
- Solve DP with true parameters
- Estimate parameters using NFXP
- Compare true vs. estimated
- Generate visualizations

## Usage Examples

### Example 1: Policy Comparison

```python
import matplotlib.pyplot as plt

# Solve for different replacement costs
replacement_costs = [8.0, 10.0, 12.0, 15.0]
policies = {}

for rc in replacement_costs:
    result = solve_dynamic_program(
        n_states, discount_factor, P,
        maintenance_costs, rc
    )
    policies[rc] = result['policy']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for rc, policy in policies.items():
    # Find first replacement state
    first_replace = np.argmax(policy)
    ax.axvline(first_replace, label=f'RC = ${rc}k', alpha=0.7)

ax.set_xlabel('State (Mileage Bin)')
ax.set_title('Replacement Threshold by Cost')
ax.legend()
plt.show()
```

### Example 2: Simulate Decision Patterns

```python
from src.simulation import ReplacementSimulator, analyze_replacement_patterns

# Create simulator with optimal policy
simulator = ReplacementSimulator(
    n_states, P,
    policy=result['policy'],
    random_state=42
)

# Simulate panel data
df_sim = simulator.simulate_panel(n_agents=50, n_periods=200)

# Analyze patterns
patterns = analyze_replacement_patterns(df_sim)
print(f"Replacement rate: {patterns['replacement_rate']:.4f}")
print(f"Mean time between replacements: {patterns['mean_time_between_replacements']:.1f}")
```

## Integration with Capital Budgeting

The optimal replacement model directly integrates with capital budgeting systems:

### Net Present Value (NPV)

The value function `V(x)` represents the expected NPV of the asset at state x under optimal management:

```
NPV(x) = V(x) / (1 - β)
```

### Replacement Timing

The model provides optimal replacement thresholds:
- **Deterministic**: Replace when `x ≥ x*` (threshold state)
- **Stochastic**: Replace with probability `P(replace | x, θ)`

### Budget Planning

Expected annual replacement expenditure for a fleet:

```
E[Replacement Cost] = N × P(replace) × RC
```

where N is fleet size and P(replace) is average replacement probability.

### Maintenance Budgeting

Expected annual maintenance cost:

```
E[Maintenance Cost] = N × Σ_x π(x) × c(x, θ)
```

where π(x) is the steady-state distribution over states.

## Applications

1. **Vehicle Fleet Management**: Bus, truck, and taxi replacement
2. **Equipment Maintenance**: Industrial machinery and HVAC systems
3. **IT Asset Management**: Server and hardware refresh cycles
4. **Infrastructure Planning**: Bridge and road maintenance schedules
5. **University Physical Assets**: Building systems and equipment replacement

## Advanced Topics

### Custom Cost Functions

```python
from src.cost_functions import CostFunction

class ExponentialCost(CostFunction):
    def evaluate(self, x):
        return self.theta[0] * np.exp(self.theta[1] * x)

cost_fn = ExponentialCost(theta=np.array([1.0, 0.05]))
```

### Custom Transition Models

```python
from src.transitions import TransitionProbability

class SeasonalTransition(TransitionProbability):
    def __init__(self, n_states, season_probs):
        super().__init__(n_states)
        self.season_probs = season_probs
    
    def get_transition_matrix(self, season):
        # Return season-specific transition matrix
        pass
```

## Performance Optimization

The implementation includes several optimizations:

- **Numba JIT compilation**: Fast value iteration for large state spaces
- **Vectorized operations**: NumPy broadcasting for efficiency
- **Sparse transitions**: Memory-efficient for large problems
- **Policy iteration**: Faster convergence for some problems

## References

**Primary Reference:**

Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher." *Econometrica*, 55(5), 999-1033.

**Related Literature:**

- Rust, J. (1994). "Structural Estimation of Markov Decision Processes." *Handbook of Econometrics*, Vol. 4.
- Aguirregabiria, V., & Mira, P. (2010). "Dynamic Discrete Choice Structural Models: A Survey." *Journal of Econometrics*, 156(1), 38-67.

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional cost function specifications
- Alternative solution algorithms (e.g., Hotz-Miller CCP)
- Heterogeneous agent models
- Continuous state space extensions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{optimal_replacement_model,
  title = {Optimal Replacement Model: Implementation of Rust (1987)},
  author = {Optimal Replacement Model Team},
  year = {2024},
  url = {https://github.com/F4b-R0D/Optimal-Replacement-Model}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.
