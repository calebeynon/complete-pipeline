#!/usr/bin/env python3
"""
Step 1: Extract Cumulative Losses and Development Factors

This script processes insurance loss triangles to extract:
1. Cumulative paid losses at different development periods
2. Loss development factors (LDFs)
3. Latest diagonal values from Mack chain ladder method

Author: Caleb
Date: February 28, 2025
"""

import Classes
import pandas as pd
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
INPUT_PATH = '../data_rep'  # Directory containing input files
OUTPUT_FILE = 'developed_and_ldfs_3-6.csv'  # Output filename

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading loss triangle data...")

# Load paid and incurred loss triangles
# These triangles contain historical loss development patterns
paid_triangles = pd.read_csv(f'{INPUT_PATH}/loss_triangles_paid.csv')
incurred_triangles = pd.read_csv(f'{INPUT_PATH}/loss_triangles_incurred.csv')

# Create Industry object that provides access to triangle calculations
ind = Classes.Industry(paid_triangles, incurred_triangles)

print(f"Loaded data for {len(ind.ygls)} company-year combinations")

# ============================================================================
# INITIALIZE COLLECTORS
# ============================================================================

# Company identifiers
years = []      # Accident year
cocodes = []    # Company code

# Development values from Mack chain ladder (latest diagonal)
# dev_0 is the most recent development, dev_8 is the oldest
dev_0 = []
dev_1 = []
dev_2 = []
dev_3 = []
dev_4 = []
dev_5 = []
dev_6 = []
dev_7 = []
dev_8 = []

# Cumulative paid losses at each development period
# Looking ahead N years from the base year to get actual paid amounts
dev_paid_0 = []
dev_paid_1 = []
dev_paid_2 = []
dev_paid_3 = []
dev_paid_4 = []
dev_paid_5 = []
dev_paid_6 = []
dev_paid_7 = []
dev_paid_8 = []

# ============================================================================
# PROCESS EACH COMPANY-YEAR
# ============================================================================

print("Processing company-year combinations...")

# Collect all loss development factors
ldf = pd.DataFrame()
processed = 0
skipped = 0

# Process each company-year combination
for l in ind.ygls:
    cum_paid = []
    yr = l[0]    # Accident year
    code = l[1]  # Company code
    
    try:
        # ----------------------------------------------------------------
        # Extract cumulative paid losses at different development periods
        # ----------------------------------------------------------------
        # We look ahead N years to see the actual paid amounts
        # This gives us the "ground truth" for model training
        cum_paid_0 = ind.loc(yr+9,code).paid.iloc[0,-1]  # 9 years later
        cum_paid_1 = ind.loc(yr+8,code).paid.iloc[0,-1]  # 8 years later
        cum_paid_2 = ind.loc(yr+7,code).paid.iloc[0,-1]  # 7 years later
        cum_paid_3 = ind.loc(yr+6,code).paid.iloc[0,-1]  # 6 years later
        cum_paid_4 = ind.loc(yr+5,code).paid.iloc[0,-1]  # 5 years later
        cum_paid_5 = ind.loc(yr+4,code).paid.iloc[0,-1]  # 4 years later
        cum_paid_6 = ind.loc(yr+3,code).paid.iloc[0,-1]  # 3 years later
        cum_paid_7 = ind.loc(yr+2,code).paid.iloc[0,-1]  # 2 years later
        cum_paid_8 = ind.loc(yr+1,code).paid.iloc[0,-1]  # 1 year later
        
        # ----------------------------------------------------------------
        # Extract loss development factors (LDFs)
        # ----------------------------------------------------------------
        # These factors are used to project losses to ultimate
        ldfs = ind.loc(yr,code).dev.ldf_.to_frame()
        
        # ----------------------------------------------------------------
        # Extract Mack chain ladder latest diagonal
        # ----------------------------------------------------------------
        # These are the developed loss estimates at each period
        latest_diag = np.array(ind.loc(yr,code).mack.latest_diagonal).flatten()
        
        # Extract values from diagonal (reading from end backwards)
        dev0 = latest_diag[-1]   # Most recent
        dev1 = latest_diag[-2]
        dev2 = latest_diag[-3]
        dev3 = latest_diag[-4]
        dev4 = latest_diag[-5]
        dev5 = latest_diag[-6]
        dev6 = latest_diag[-7]
        dev7 = latest_diag[-8]
        dev8 = latest_diag[-9]   # Oldest
        
    except Exception as e:
        # Skip if data is incomplete for this company-year
        skipped += 1
        continue
    
    # ----------------------------------------------------------------
    # Store extracted values
    # ----------------------------------------------------------------
    
    # Append loss development factors
    ldf = pd.concat([ldf,ldfs],axis=0,ignore_index=True)
    
    # Store identifiers
    years.append(yr)
    cocodes.append(code)
    
    # Store developed values from Mack method
    dev_0.append(dev0)
    dev_1.append(dev1)
    dev_2.append(dev2)
    dev_3.append(dev3)
    dev_4.append(dev4)
    dev_5.append(dev5)
    dev_6.append(dev6)
    dev_7.append(dev7)
    dev_8.append(dev8)
    
    # Store actual cumulative paid amounts
    dev_paid_0.append(cum_paid_0)
    dev_paid_1.append(cum_paid_1)
    dev_paid_2.append(cum_paid_2)
    dev_paid_3.append(cum_paid_3)
    dev_paid_4.append(cum_paid_4)
    dev_paid_5.append(cum_paid_5)
    dev_paid_6.append(cum_paid_6)
    dev_paid_7.append(cum_paid_7)
    dev_paid_8.append(cum_paid_8)
    
    processed += 1
    if processed % 100 == 0:
        print(f"  Processed {processed} records...")

print(f"Processing complete: {processed} processed, {skipped} skipped")
# ============================================================================
# CREATE OUTPUT DATAFRAME
# ============================================================================

print("Creating output dataframe...")

# Create main dataframe with all extracted values
developed_df = pd.DataFrame({
    # Identifiers
    'year': years,
    'cocode': cocodes,
    
    # Actual cumulative paid losses at each development period
    'cumulative_paid_0': dev_paid_0,  # 9 years of development
    'cumulative_paid_1': dev_paid_1,  # 8 years of development
    'cumulative_paid_2': dev_paid_2,  # 7 years of development
    'cumulative_paid_3': dev_paid_3,  # 6 years of development
    'cumulative_paid_4': dev_paid_4,  # 5 years of development
    'cumulative_paid_5': dev_paid_5,  # 4 years of development
    'cumulative_paid_6': dev_paid_6,  # 3 years of development
    'cumulative_paid_7': dev_paid_7,  # 2 years of development
    'cumulative_paid_8': dev_paid_8,  # 1 year of development
    
    # Mack chain ladder developed values
    'dev_0': dev_0,  # Most recent diagonal
    'dev_1': dev_1,
    'dev_2': dev_2,
    'dev_3': dev_3,
    'dev_4': dev_4,
    'dev_5': dev_5,
    'dev_6': dev_6,
    'dev_7': dev_7,
    'dev_8': dev_8   # Oldest diagonal
})

# Combine with loss development factors
result_df = pd.concat([developed_df, ldf], axis=1)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"Saving results to {OUTPUT_FILE}...")
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nComplete! Saved {len(result_df)} records")
print(f"Output contains:")
print(f"  - Cumulative paid losses for 9 development periods")
print(f"  - Chain ladder developed values for 9 periods")
print(f"  - Loss development factors")














