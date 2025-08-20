#!/usr/bin/env python3
"""
Step 7: Add Chain Ladder Increase Indicator

This script adds a binary feature indicating whether the chain ladder
estimate increased from the previous period. This captures reserve
strengthening events - when a company realizes their initial loss
estimates were too low and needs to increase reserves.

Reserve strengthening is a critical indicator in insurance:
- It suggests initial estimates were inadequate
- May indicate deteriorating claim experience
- Often correlates with specific business or economic conditions

Author: Caleb
Date: 2025
"""

import pandas as pd
import tri_classes_v2 as tc

# ============================================================================
# CONFIGURATION
# ============================================================================

# FILE PATHS
TRIANGLE_PAID = 'loss_triangles_paid.csv'
TRIANGLE_INCURRED = 'loss_triangles_incurred.csv'
INPUT_FILE = 'lag_data_0416.csv'  # From Step 6 (with peer averages)
OUTPUT_FILE = 'lag_data_0419.csv'  # Final dataset for neural network

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading loss triangles...")
p = pd.read_csv(TRIANGLE_PAID)
i = pd.read_csv(TRIANGLE_INCURRED)
ind = tc.Industry(p, i)

print("Loading feature dataset...")
X = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(X)} records with {len(X.columns)} features")

# ============================================================================
# EXTRACT CHAIN LADDER INCREASE INDICATOR
# ============================================================================

print("Extracting chain ladder increase indicators...")

# For each company-year, check if chain ladder estimate increased
# This is calculated by the Industry class comparing successive periods
X['CL_increase'] = X.apply(
    lambda row: ind.loc(row['year'], row['cocode']).CL_increase, 
    axis=1
)

print("Converting boolean values to binary (0/1)...")

# Convert boolean values to binary for neural network
# True -> 1 (reserve strengthening occurred)
# False -> 0 (no reserve strengthening)
X['CL_increase'] = X['CL_increase'].apply(
    lambda x: 1 if x == True else (0 if x == False else x)
)

# Check distribution of the indicator
increases = X['CL_increase'].sum()
print(f"  {increases} companies ({increases/len(X)*100:.1f}%) had reserve strengthening")

# ============================================================================
# SAVE FINAL DATASET
# ============================================================================

print(f"Saving final dataset to {OUTPUT_FILE}...")
X.to_csv(OUTPUT_FILE, index=False)

print(f"\nComplete! Created final dataset with {len(X.columns)} features")
print("Key features include:")
print("  - Company characteristics and financials")
print("  - Loss development factors and chain ladder estimates")
print("  - Full triangle features (ft_0 to ft_54)")
print("  - Peer averages (avg_ft_0 to avg_ft_54)")
print("  - Economic indicators")
print("  - Chain ladder increase indicator")
print("\nDataset ready for neural network training (Step 8)")







