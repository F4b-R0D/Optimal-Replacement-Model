# Data Directory

This directory contains sample datasets for the Rust (1987) optimal replacement model.

## Files

- : Small dataset (500 observations) for quick testing
- : Medium dataset (2000 observations) for examples
- : Large dataset (10000 observations) for estimation

## Generating Data

To generate sample datasets, run:

```bash
cd automation
python generate_data.py
```

## Data Format

Datasets follow this structure:

| Column | Description |
|--------|-------------|
| bus_id | Unique identifier for each bus |
| period | Time period (month) |
| mileage_bin | Discretized mileage state (0-89) |
| decision | Decision (0=keep, 1=replace) |

## True Parameters

Synthetic datasets are generated with known parameters:
- Maintenance cost: c(x) = 0.0 + 0.001 * x (linear)
- Replacement cost: RC = 11.7 (thousand dollars)
- Discount factor: β = 0.9999 (monthly)
- Transition probabilities: [0.349, 0.639, 0.012] for increments [0, 1, 2]

