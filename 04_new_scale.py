#!/usr/bin/env python3
"""
Step 4: Feature Scaling and Residual Creation

This script performs critical preprocessing for neural network training:
1. Scales numerical features using MinMaxScaler and StandardScaler
2. Creates residuals (actual losses - chain ladder estimates)
3. Applies winsorization to handle outliers in target variables

The residuals represent what the neural network needs to learn - the
difference between actual losses and chain ladder predictions.

Author: Caleb
Date: 2025
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def winsorize_df(df, lower=0.01, upper=0.99):
    """
    Apply winsorization to handle outliers in numerical data.
    
    Winsorization caps extreme values at specified percentiles to reduce
    the impact of outliers on model training.
    
    Args:
        df: DataFrame to winsorize
        lower: Lower percentile for capping (default 1%)
        upper: Upper percentile for capping (default 99%)
    
    Returns:
        DataFrame with winsorized values
    """
    df_winsorized = df.copy()
    
    # Process each numerical column
    for col in df_winsorized.select_dtypes(include=[np.number]):
        # Calculate percentile bounds
        lower_bound = np.percentile(df_winsorized[col].dropna(), lower * 100)
        upper_bound = np.percentile(df_winsorized[col].dropna(), upper * 100)
        # Cap values at the bounds
        df_winsorized[col] = np.clip(df_winsorized[col], lower_bound, upper_bound)
    
    return df_winsorized

# ============================================================================
# CONFIGURATION
# ============================================================================

# FILE PATHS
INPUT_PATH = 'data_rep'
OUTPUT_PATH = 'data_rep'

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data from previous step...")
X = pd.read_csv(f'{OUTPUT_PATH}/data0306_with_cldev.csv')

# ============================================================================
# SELECT FEATURES
# ============================================================================

print("Selecting features for neural network...")

# Define which columns to keep
# Note: Some columns (ft_*, avg_ft_*) will be added in later steps
columns_to_keep = ['cocode', 'year'] + \
                  [f'cumulative_paid_{i}' for i in range(9)] + \
                  [f'cl_dev_{i}' for i in range(9)] + \
                  [f'X{i}.{i+12}' for i in range(12,120,12)] + \
                  [f'dev_{i}' for i in range(9)] + \
                  [f'ft_{i}' for i in range(55)] + \
                  [f'avg_ft_{i}' for i in range(55)] + \
                  ['wPOP', 'wEMP', 'wINC']

# Keep only columns that exist in current dataset
# (ft_* and avg_ft_* will be added in steps 5 and 6)
existing_columns = [col for col in columns_to_keep if col in X.columns]
X = X[existing_columns]

print(f"Selected {len(existing_columns)} features")

# ============================================================================
# CREATE RESIDUALS (TARGET VARIABLES)
# ============================================================================

print("Creating residual targets...")

# Calculate residuals for each development period
# Residual = Actual cumulative paid - Chain ladder estimate
# This is what the neural network will learn to predict
for i in range(9):
    X[f'residual_{i}'] = X[f'cumulative_paid_{i}'] - X[f'cl_dev_{i}']

# Extract residuals as target variables
y = X[[f'residual_{i}' for i in range(9)]]

# Remove residuals from feature set
X = X.drop(columns=[f'residual_{i}' for i in range(9)])

# Keep unscaled version for reference
X_unscaled = X.copy()

# ============================================================================
# SCALE FEATURES
# ============================================================================

print("Scaling features using MinMaxScaler...")

# Scale loss development factors (age-to-age factors)
# These represent period-to-period growth rates
scaler = MinMaxScaler()
X[[f'X{i}.{i+12}' for i in range(12,120,12)]] = scaler.fit_transform(X[[f'X{i}.{i+12}' for i in range(12,120,12)]])

# Scale development values from Mack method
devscaler = MinMaxScaler()
X[[f'dev_{i}' for i in range(9)]] = devscaler.fit_transform(X[[f'dev_{i}' for i in range(9)]])

# Scale chain ladder estimates
# Important: Save this scaler for inverse transformation later
clscaler = MinMaxScaler()
X[[f'cl_dev_{i}' for i in range(9)]] = clscaler.fit_transform(X[[f'cl_dev_{i}' for i in range(9)]])

# Scale economic indicators
# Each gets its own scaler to preserve relative magnitudes
popscaler = MinMaxScaler()
X[['wPOP']] = popscaler.fit_transform(X[['wPOP']])

empscaler = MinMaxScaler()
X[['wEMP']] = empscaler.fit_transform(X[['wEMP']])

incscaler = MinMaxScaler()
X[['wINC']] = incscaler.fit_transform(X[['wINC']])

print("Feature scaling complete")

# ============================================================================
# SCALE TARGET VARIABLES (RESIDUALS)
# ============================================================================

print("Processing target variables...")

# Apply winsorization to handle extreme outliers
# This prevents a few extreme losses from dominating the training
ywin = winsorize_df(y, lower=0.01, upper=0.99)

# Standardize residuals to zero mean and unit variance
# This helps with neural network convergence
yscaler = StandardScaler()
y_s = yscaler.fit_transform(ywin)
y_s = pd.DataFrame(y_s, columns=y.columns, index=y.index)

print("Target scaling complete")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("Saving scaled data and scalers...")

# Save SCALED version as lag_data_scaled_0316.csv for the next step
# This is the primary output used by step 5 (adding full triangle)
X.to_csv('lag_data_scaled_0316.csv', index=False)
print("  Saved lag_data_scaled_0316.csv for next pipeline step")

# Also save the standard outputs for alternative workflows
X.to_csv(f'{OUTPUT_PATH}/X_scaled.csv', index=False)
y_s.to_csv(f'{OUTPUT_PATH}/y_scaled.csv', index=False)
print("  Saved X_scaled.csv and y_scaled.csv")

# Save scalers for inverse transformation
# These are critical for converting predictions back to original scale
with open(f'{OUTPUT_PATH}/yscaler.pkl', 'wb') as f:
    pickle.dump(yscaler, f)

with open(f'{OUTPUT_PATH}/clscaler.pkl', 'wb') as f:
    pickle.dump(clscaler, f)
print("  Saved scalers for inverse transformation")

print(f"\nComplete! Processed {len(X)} records with {len(X.columns)} features")
print("Ready for Step 5: Adding full triangle features")
