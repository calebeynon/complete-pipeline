# Complete Insurance Loss Reserving Pipeline

## Overview
This folder contains the complete pipeline for insurance loss reserving using neural networks, this code can be used to replicate the results found in my current working paper: A Neural Network Approach to Loss Reserving.

If attempting to replicate results, follow the numbered order of the scripts when running.
If viewing this as an attempt to understand the code, or as a sample of my programming abilities, I recommend starting at `08_neural_network.py` as it provides the model and usage of the data pipeline. From there, view `Classes.py` as a way to understand the storage structure for the data in this project.

Note: AI was used to assist with documentation of this project
Note: The code used to generate the graphs found in the paper is included, but it is still a work in progress.

## Prerequisites

### Required Files in `data_rep/` directory:
- `loss_triangles_paid.csv`
- `loss_triangles_incurred.csv`
- `Rprocessed AMB`
- `Rprocessed_assets`
- `Rprocessed_BEA`
- `Rprocessed_demo`
- `Rprocessed_Page14`
- `Rprocessed_uie_pw`

## Pipeline Execution Order

### Step 1: `01_cumulative_losses.py`
**Purpose**: Extract cumulative losses and development factors from loss triangles
- **Inputs**: 
  - `loss_triangles_paid.csv`
  - `loss_triangles_incurred.csv`
- **Outputs**: 
  - `developed_and_ldfs_3-6.csv`

### Step 2: `02_variable_creation.R`
**Purpose**: Create firm-specific variables and economic indicators
- **Inputs**:
  - All `Rprocessed_*` files
  - `developed_and_ldfs_3-6.csv` (from Step 1)
- **Outputs**:
  - `data1023.csv` (intermediate - company data without economic variables)
  - `data0302.csv` (intermediate - company data with economic variables)
  - `data0306.csv` (final merged dataset)

### Step 3: `03_add_cldev_columns.py`
**Purpose**: Add chain ladder development estimates
- **Inputs**:
  - `data0306.csv` (from Step 2)
  - `loss_triangles_paid.csv`
  - `loss_triangles_incurred.csv`
- **Outputs**:
  - `data0306_with_cldev.csv`

### Step 4: `04_new_scale.py`
**Purpose**: Scale features and create lag data base file
- **Inputs**:
  - `data0306_with_cldev.csv` (from Step 3)
- **Outputs**:
  - `lag_data_scaled_0316.csv` (scaled features for next steps)
  - `X_scaled.csv` (for alternative workflows)
  - `y_scaled.csv` (scaled target residuals)
  - `yscaler.pkl`, `clscaler.pkl` (scalers for inverse transformation)

### Step 5: `05_adding_full_triangle.py`
**Purpose**: Add full triangle features (ft_0 to ft_54)
- **Inputs**:
  - `lag_data_scaled_0316.csv` (from Step 4)
  - `loss_triangles_paid.csv`
  - `loss_triangles_incurred.csv`
- **Outputs**:
  - `lag_data_scaled_0407.csv`

### Step 6: `06_lag_data_creation.R`
**Purpose**: Create peer averages by company type
- **Inputs**:
  - `lag_data_scaled_0407.csv` (from Step 5)
- **Outputs**:
  - `lag_data_0416.csv`

### Step 7: `07_CL_increase_add.py`
**Purpose**: Add chain ladder increase indicator
- **Inputs**:
  - `lag_data_0416.csv` (from Step 6)
  - `loss_triangles_paid.csv`
  - `loss_triangles_incurred.csv`
- **Outputs**:
  - `lag_data_0419.csv` (final dataset for neural network)

### Step 8: `08_neural_network.py`
**Purpose**: Train neural network models
- **Inputs**:
  - `lag_data_0419.csv` (from Step 7)
- **Process**:
  - Performs its own feature scaling
  - Creates residuals (actual - chain ladder)
  - Trains models with different feature sets
- **Outputs**:
  - Trained models and predictions

## Running the Pipeline

Execute each script in order:
```bash
# Step 1
python 01_cumulative_losses.py

# Step 2
Rscript 02_variable_creation.R

# Step 3
python 03_add_cldev_columns.py

# Step 4
python 04_new_scale.py

# Step 5
python 05_adding_full_triangle.py

# Step 6
Rscript 06_lag_data_creation.R

# Step 7
python 07_CL_increase_add.py

# Step 8
python 08_neural_network.py
```

## Data Flow Summary

```
Loss Triangles → Cumulative Losses → Company Variables → Chain Ladder Estimates 
→ Scaled Features → Full Triangle Features → Peer Averages → CL Increase Flag 
→ Neural Network Training
```

## Key Features Created

1. **Development Factors**: `dev_0` to `dev_8`, `X12.24` to `X108.120`
2. **Chain Ladder Estimates**: `cl_dev_0` to `cl_dev_8`
3. **Full Triangle Features**: `ft_0` to `ft_54` (55 features)
4. **Peer Averages**: `avg_ft_0` to `avg_ft_54`
5. **Economic Indicators**: `wPOP`, `wEMP`, `wINC`
6. **CL Increase Flag**: Binary indicator for reserve strengthening

## Notes

- The pipeline progressively builds features through multiple steps
- "scaled" in filenames refers to MinMaxScaler/StandardScaler transformations
- The neural network (Step 8) does additional scaling internally
- All file paths assume execution from the `complete_pipeline` directory