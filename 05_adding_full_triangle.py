#!/usr/bin/env python3
"""
Step 5: Add Full Triangle Features

This script adds the complete loss triangle as features for the neural network.
The triangle is "flattened" into 55 features (ft_0 to ft_54) representing
the full 10x10 upper triangle of loss development.

These features provide the neural network with the complete historical
loss development pattern for each company-year, enabling it to learn
complex patterns beyond what simple chain ladder can capture.

Author: Caleb
Date: 2025
"""

import pandas as pd
import numpy as np
import tri_classes_v2 as tc
import chainladder as cl

# ============================================================================
# CONFIGURATION
# ============================================================================

# FILE PATHS
INPUT_FILE = 'lag_data_scaled_0316.csv'  # From Step 4 (scaled features)
TRIANGLE_PAID = 'Loss_triangles_paid.csv'
TRIANGLE_INCURRED = 'loss_triangles_incurred.csv'
OUTPUT_FILE = 'lag_data_scaled_0407.csv'

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading scaled features from previous step...")
df = pd.read_csv(INPUT_FILE)  # Note: this includes scaled data, not all raw
print(f"Loaded {len(df)} records with {len(df.columns)} features")

print("Loading loss triangles...")
ind = tc.Industry(pd.read_csv(TRIANGLE_PAID), pd.read_csv(TRIANGLE_INCURRED))

# ============================================================================
# EXTRACT FULL TRIANGLE FEATURES
# ============================================================================

print("Extracting full triangle features for each company-year...")

# Get company-year pairs to process
tups = df[['cocode', 'year']]

# Initialize collector for triangle features
ft = pd.DataFrame()
processed = 0

# Process each company-year
for row in tups.itertuples(index=True):
    try:
        # Extract the paid loss triangle for this company-year
        # The triangle is a 10x10 matrix but we only use the upper triangle
        # (55 values total: 10+9+8+7+6+5+4+3+2+1)
        triangle_array = ind.loc(row.year, row.cocode).retTri['paid'].to_numpy()
        
        # Flatten the triangle into a single row (1x55)
        # This preserves all development information in feature form
        temp = pd.DataFrame(triangle_array.reshape(1, -1))
        
        # Add identifiers
        temp['cocode'] = row.cocode
        temp['year'] = row.year
        
        # Append to collection
        ft = pd.concat([ft, temp], axis=0).reset_index(drop=True)
        
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed} records...")
            
    except Exception as e:
        print(f"  Warning: Could not extract triangle for cocode {row.cocode}, year {row.year}")
        continue

print(f"Extracted triangles for {len(ft)} company-years")

# Rename columns to ft_0 through ft_54 plus identifiers
# ft_0 represents the earliest/smallest value, ft_54 the latest/largest
ft.columns = [f'ft_{i}' for i in range(55)] + ['cocode', 'year']

# ============================================================================
# MERGE WITH EXISTING FEATURES
# ============================================================================

print("Merging triangle features with existing dataset...")

# Merge on company-year identifiers
df_final = pd.merge(df, ft, on=['cocode', 'year'])

print(f"Final dataset has {len(df_final)} records with {len(df_final.columns)} features")
print(f"Added 55 new triangle features (ft_0 to ft_54)")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

print(f"Saving to {OUTPUT_FILE}...")
df_final.to_csv(OUTPUT_FILE, index=False)

print("\nComplete! Ready for Step 6: Creating peer averages")
print("The full triangle features provide the neural network with:")
print("  - Complete historical loss development patterns")
print("  - Raw data to identify company-specific trends")
print("  - Information beyond simple development factors")


