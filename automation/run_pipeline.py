"""
Automation pipeline for running complete analysis workflow.

This script demonstrates the full workflow from data generation to estimation
and visualization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.simulation import create_rust_dataset, analyze_replacement_patterns
from src.transitions import create_transition_model
from src.cost_functions import create_cost_function
from src.dp import solve_dynamic_program
from src.nfxp import NFXPEstimator, linear_cost_wrapper


def run_pipeline(save_figures: bool = True):
    """
    Run complete analysis pipeline.
    
    Steps:
    1. Generate synthetic data
    2. Solve DP with true parameters
    3. Estimate parameters using NFXP
    4. Compare true vs estimated
    5. Generate visualizations
    """
    
    print("=" * 60)
    print("RUST (1987) OPTIMAL REPLACEMENT MODEL - FULL PIPELINE")
    print("=" * 60)
    
    # Configuration
    n_states = 90
    discount_factor = 0.9999
    true_maintenance_params = np.array([0.0, 0.001])
    true_replacement_cost = 11.7
    
    # Step 1: Generate data
    print("\n1. Generating synthetic data...")
    df = create_rust_dataset(sample_size=5000, n_states=n_states, random_state=42)
    print(f"   Generated {len(df)} observations")
    
    # Analyze patterns
    patterns = analyze_replacement_patterns(df)
    print(f"   Replacement rate: {patterns['replacement_rate']:.4f}")
    print(f"   Mean mileage: {df['mileage_bin'].mean():.2f}")
    
    # Step 2: Solve DP with true parameters
    print("\n2. Solving DP with true parameters...")
    trans_model = create_transition_model("discrete", n_states=n_states)
    P = trans_model.get_transition_matrix()
    
    cost_fn = create_cost_function("linear", theta=true_maintenance_params)
    state_grid = np.arange(n_states)
    maintenance_costs = cost_fn.evaluate(state_grid)
    
    result = solve_dynamic_program(
        n_states, discount_factor, P,
        maintenance_costs, true_replacement_cost,
        verbose=False
    )
    print(f"   Converged in {result['iterations']} iterations")
    
    # Step 3: Estimate parameters
    print("\n3. Estimating parameters using NFXP...")
    estimator = NFXPEstimator(n_states, discount_factor, P)
    
    # Prepare data
    states = df['mileage_bin'].values
    decisions = df['decision'].values
    
    # Initial guess
    theta_init = np.array([0.0, 0.002, 10.0])
    
    est_result = estimator.estimate(
        states, decisions, linear_cost_wrapper,
        theta_init=theta_init, verbose=True
    )
    
    # Step 4: Compare results
    print("\n4. Comparing true vs estimated parameters:")
    print(f"   True parameters:      {np.append(true_maintenance_params, true_replacement_cost)}")
    print(f"   Estimated parameters: {est_result['theta']}")
    if est_result['standard_errors'] is not None:
        print(f"   Standard errors:      {est_result['standard_errors']}")
    
    # Step 5: Generate visualizations
    if save_figures:
        print("\n5. Generating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Value function
        ax = axes[0, 0]
        ax.plot(state_grid, result['value_function'], label='True parameters')
        pred = estimator.predict(cost_function=linear_cost_wrapper)
        ax.plot(state_grid, pred['value_function'], '--', label='Estimated parameters')
        ax.set_xlabel('State (Mileage Bin)')
        ax.set_ylabel('Value Function')
        ax.set_title('Value Function Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Maintenance cost function
        ax = axes[0, 1]
        true_costs = cost_fn.evaluate(state_grid)
        est_cost_fn = create_cost_function("linear", theta=est_result['theta'][:-1])
        est_costs = est_cost_fn.evaluate(state_grid)
        ax.plot(state_grid, true_costs, label='True cost')
        ax.plot(state_grid, est_costs, '--', label='Estimated cost')
        ax.set_xlabel('State (Mileage Bin)')
        ax.set_ylabel('Maintenance Cost')
        ax.set_title('Cost Function Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Replacement probability
        ax = axes[1, 0]
        ax.plot(state_grid, pred['choice_probabilities'], label='Estimated model')
        ax.axhline(y=patterns['replacement_rate'], color='r', 
                  linestyle=':', label='Overall replacement rate')
        ax.set_xlabel('State (Mileage Bin)')
        ax.set_ylabel('Replacement Probability')
        ax.set_title('Replacement Probability by State')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: State distribution
        ax = axes[1, 1]
        state_counts = df['mileage_bin'].value_counts().sort_index()
        ax.bar(state_counts.index, state_counts.values / len(df), alpha=0.7)
        ax.set_xlabel('State (Mileage Bin)')
        ax.set_ylabel('Frequency')
        ax.set_title('Observed State Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'pipeline_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Saved figure to {output_path}")
        plt.close()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        'data': df,
        'true_params': np.append(true_maintenance_params, true_replacement_cost),
        'estimated_params': est_result['theta'],
        'estimation_results': est_result,
        'dp_results': result,
        'patterns': patterns
    }


if __name__ == "__main__":
    results = run_pipeline(save_figures=True)
