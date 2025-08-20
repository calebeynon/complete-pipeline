#!/usr/bin/env python3
"""
Step 3: Add Chain Ladder Development Estimates

This script adds chain ladder ultimate loss estimates to the dataset.
These estimates serve as the baseline predictions that the neural network
will learn to improve upon through residual learning.

Author: Caleb
Date: 2025
"""

import pandas as pd
import numpy as np
import sys

# Add current directory to path for Classes module
sys.path.append('.')
from Classes import Industry

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH = 'data_rep'
OUTPUT_PATH = 'data_rep'

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading triangle data for chain ladder calculations...")

# Load loss triangles
paid = pd.read_csv(f'{INPUT_PATH}/loss_triangles_paid.csv')
incurred = pd.read_csv(f'{INPUT_PATH}/loss_triangles_incurred.csv')

# Create Industry object for accessing chain ladder calculations
ind = Industry(paid, incurred)

print("Loading company data from previous step...")

# Load the dataset created in Step 2 (with company variables)
main_df = pd.read_csv(f'{OUTPUT_PATH}/data0306.csv')
print(f"Loaded {len(main_df)} company-year records")

# ============================================================================
# EXTRACT CHAIN LADDER ESTIMATES
# ============================================================================

print("Calculating chain ladder estimates for each company-year...")

# Get company-year identifiers
years = main_df['year'].values
cocodes = main_df['cocode'].values

# Collect chain ladder development estimates
cldev_rows = []
processed = 0
skipped = 0

for year, cocode in zip(years, cocodes):
    try:
        # ----------------------------------------------------------------
        # Get Mack chain ladder ultimate loss estimates
        # ----------------------------------------------------------------
        # The latest diagonal contains the chain ladder projections
        # for each development period
        ult = ind.loc(year, cocode).mack.latest_diagonal
        
        # Convert to numpy array and flatten
        ult = np.array(ult).flatten()
        
        # ----------------------------------------------------------------
        # Standardize to 10 development periods
        # ----------------------------------------------------------------
        if len(ult) < 10:
            # Pad with NaN if fewer than 10 development periods
            ult = np.pad(ult, (10-len(ult), 0), constant_values=np.nan)
        elif len(ult) > 10:
            # Take only the last 10 if more than 10 periods
            ult = ult[-10:]
        
        # Create dictionary with cl_dev_0 through cl_dev_9
        # cl_dev_0 is the most recent, cl_dev_9 is the oldest
        cldev_row = {f'cl_dev_{i}': ult[-(i+1)] for i in range(10)}
        processed += 1
        
    except Exception as e:
        # If chain ladder calculation fails, fill with NaN
        cldev_row = {f'cl_dev_{i}': np.nan for i in range(10)}
        skipped += 1
    
    cldev_rows.append(cldev_row)
    
    if processed % 100 == 0:
        print(f"  Processed {processed} records...")

print(f"Chain ladder calculation complete: {processed} successful, {skipped} failed")

# ============================================================================
# MERGE WITH MAIN DATASET
# ============================================================================

print("Merging chain ladder estimates with main dataset...")

# Create dataframe with chain ladder estimates
cldev_df = pd.DataFrame(cldev_rows)
cldev_df['year'] = years
cldev_df['cocode'] = cocodes

# Merge with main dataset
merged = pd.merge(main_df, cldev_df, on=['year', 'cocode'], how='inner')

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_file = f'{OUTPUT_PATH}/data0306_with_cldev.csv'
print(f"Saving to {output_file}...")
merged.to_csv(output_file, index=False)

print(f"\nComplete! Saved {len(merged)} records")
print(f"Added 10 chain ladder estimate columns (cl_dev_0 through cl_dev_9)")
print("These estimates will serve as baseline predictions for the neural network")