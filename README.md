# AI-Assisted Surrogate Modeling for Engineering Simulation Acceleration

## Overview

This project develops and validates machine learning surrogate models to replace computationally expensive finite element simulations. 
Gaussian Process Regression and Artificial Neural Networks were trained on Latin Hypercube sampled simulation data to enable rapid probabilistic reliability analysis.
The framework reduces Monte Carlo evaluation time by several orders of magnitude while maintaining high predictive accuracy.

## Problem Statement

High-fidelity finite element simulations are computationally expensive when used in uncertainty quantification or Monte Carlo analysis. 
The objective of this project is to construct validated surrogate models that approximate simulation outputs and enable fast probabilistic design evaluation.

## Methodology

1. Latin Hypercube Sampling (100 samples) for parameter space exploration
2. Automated finite element simulations (Abaqus)
3. Dataset construction and normalization
4. Surrogate model training:
   - Gaussian Process Regression (Kriging)
   - Artificial Neural Network (Feedforward)
5. Model validation using RMSE and R²
6. Monte Carlo reliability analysis (100,000 samples)
7. Extraction of critical design thresholds

## Results

- Gaussian Process: R² = 0.999999, RMSE < 0.05 MPa
- ANN: R² = 0.9999
- >10⁴× computational speedup compared to direct FEM Monte Carlo
- Reliable identification of safe and burst operating thresholds

## Repository Structure

├── data_sample.csv  
├── surrogate_model.py  
├── monte_carlo_reliability.py  
├── figures/  
└── README.md  

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run surrogate training:
   python surrogate_model.py

3. Run reliability analysis:
   python monte_carlo_reliability.py

## Tools & Technologies

- Python (NumPy, scikit-learn, Matplotlib)
- MATLAB
- Abaqus
- Monte Carlo Simulation
- Latin Hypercube Sampling

## Future Work

- Sensitivity analysis (Sobol indices)
- Adaptive sampling strategy
- Comparison with FORM reliability method
