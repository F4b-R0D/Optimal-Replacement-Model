"""
Script to generate sample datasets for the Rust (1987) model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.simulation import create_rust_dataset


def generate_sample_data():
    """Generate and save sample datasets."""
    
    print("Generating sample Rust (1987) bus replacement data...")
    
    # Small dataset for testing
    df_small = create_rust_dataset(sample_size=500, random_state=42)
    df_small.to_csv('../data/sample_small.csv', index=False)
    print(f"  - Small dataset: {len(df_small)} observations")
    
    # Medium dataset for examples
    df_medium = create_rust_dataset(sample_size=2000, random_state=43)
    df_medium.to_csv('../data/sample_medium.csv', index=False)
    print(f"  - Medium dataset: {len(df_medium)} observations")
    
    # Large dataset for estimation
    df_large = create_rust_dataset(sample_size=10000, random_state=44)
    df_large.to_csv('../data/sample_large.csv', index=False)
    print(f"  - Large dataset: {len(df_large)} observations")
    
    # Print summary statistics
    print("\nSummary statistics (medium dataset):")
    print(f"  Replacement rate: {df_medium['decision'].mean():.4f}")
    print(f"  Mean mileage: {df_medium['mileage_bin'].mean():.2f}")
    print(f"  Number of buses: {df_medium['bus_id'].nunique()}")
    
    print("\nDatasets saved to data/ directory")


if __name__ == "__main__":
    generate_sample_data()
