#!/usr/bin/env python3
"""
Step 8: Neural Network Training with Transfer Learning

This script trains neural networks to predict insurance loss residuals
using a transfer learning approach. The models learn to improve upon
chain ladder estimates by predicting the residual (error) between
actual losses and chain ladder predictions.

Key innovations:
1. Residual learning - predict corrections to chain ladder, not raw losses
2. Transfer learning - reuse base model weights when adding new features
3. Multiple feature sets - compare different combinations of predictors

Author: Caleb
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def winsorize_df(df, lower=0.01, upper=0.99):
    """
    Apply winsorization to handle outliers in numerical data.
    Caps extreme values at specified percentiles.
    """
    df_winsorized = df.copy()
    
    for col in df_winsorized.select_dtypes(include=[np.number]):
        lower_bound = np.percentile(df_winsorized[col].dropna(), lower * 100)
        upper_bound = np.percentile(df_winsorized[col].dropna(), upper * 100)
        df_winsorized[col] = np.clip(df_winsorized[col], lower_bound, upper_bound)
    
    return df_winsorized

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("Loading final dataset...")
X = pd.read_csv('lag_data_0419.csv')  # Note: includes some scaled data from earlier steps

# Remove columns 8-87 (specific columns not needed for this analysis)
X = X.drop(X.columns[8:87], axis=1)

print(f"Loaded {len(X)} records with initial {len(X.columns)} features")

# ============================================================================
# CREATE RESIDUAL TARGETS
# ============================================================================

print("Creating residual targets...")

# Calculate residuals: actual cumulative paid - chain ladder estimate
# These residuals are what the neural network learns to predict
for i in range(9):
    X[f'residual_{i}'] = X[f'cumulative_paid_{i}'] - X[f'cl_dev_{i}']

# Extract residuals as target variables
y = X[[f'residual_{i}' for i in range(9)]]

# Remove residuals and extra column from features
X = X.drop(columns=[f'residual_{i}' for i in range(9)] + ['cl_dev_9'])

print(f"Created {y.shape[1]} residual targets")

# ============================================================================
# SCALE FEATURES
# ============================================================================

print("Scaling features...")

# Scale different feature groups with appropriate scalers
# Loss development factors (age-to-age factors)
scaler = MinMaxScaler()
X[[f'X{i}.{i+12}' for i in range(12,120,12)]] = scaler.fit_transform(
    X[[f'X{i}.{i+12}' for i in range(12,120,12)]]
)

# Development values from Mack method
devscaler = MinMaxScaler()
X[[f'dev_{i}' for i in range(9)]] = devscaler.fit_transform(
    X[[f'dev_{i}' for i in range(9)]]
)

# Chain ladder estimates (save scaler for inverse transformation)
clscaler = MinMaxScaler()
X[[f'cl_dev_{i}' for i in range(9)]] = clscaler.fit_transform(
    X[[f'cl_dev_{i}' for i in range(9)]]
)

# Full triangle features
ftscaler = MinMaxScaler()
X[[f'ft_{i}' for i in range(55)]] = ftscaler.fit_transform(
    X[[f'ft_{i}' for i in range(55)]]
)

# Peer average features
lagscaler = MinMaxScaler()
X[[f'avg_ft_{i}' for i in range(55)]] = lagscaler.fit_transform(
    X[[f'avg_ft_{i}' for i in range(55)]]
)

# ============================================================================
# SCALE TARGET VARIABLES
# ============================================================================

print("Processing target variables...")

# Winsorize to handle extreme outliers
ywin = winsorize_df(y, lower=0.01, upper=0.99)

# Standardize residuals for neural network training
yscaler = StandardScaler()
y_s = yscaler.fit_transform(ywin)
y_s = pd.DataFrame(y_s, columns=y.columns, index=y.index)

# Function to reverse the standardization
def unscaled_pred(df, yscaler=yscaler):
    """Reverse StandardScaler transformation"""
    std = yscaler.scale_
    mean = yscaler.mean_
    return df * std + mean

# ============================================================================
# NEURAL NETWORK TRAINING FUNCTIONS
# ============================================================================

# Configure early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',        
    patience=10,               # Wait 10 epochs without improvement
    restore_best_weights=True  # Restore best weights when stopping
)

def train_nn(Xi, yi, cols=None):
    """
    Train base neural network model.
    
    Args:
        Xi: Feature dataframe
        yi: Target dataframe (scaled residuals)
        cols: List of feature columns to use
    
    Returns:
        model: Trained Keras model
        results: Dictionary of mean absolute errors by development period
    """
    # Select specified columns plus identifiers and chain ladder estimates
    Xi = Xi[cols + ['cocode', 'year'] + [f'cl_dev_{i}' for i in range(9)]]
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=0.2)
    
    # Build neural network architecture
    model = Sequential()
    model.add(Input(shape=(X_train.drop(columns=['cocode','year'] + 
                                        [f'cl_dev_{i}' for i in range(9)]).shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(9))  # 9 outputs (one per development period)
    
    # Compile with MAE loss (robust to outliers)
    model.compile(optimizer=Adam(learning_rate=1e-05, clipnorm=1.0), 
                 loss='mae', 
                 metrics=['mae'])
    
    # Train the model
    model.fit(X_train.drop(columns=['year','cocode'] + [f'cl_dev_{i}' for i in range(9)]),
             y_train,
             epochs=100,
             batch_size=8,
             validation_split=0.2,
             callbacks=[early_stop])
    
    # ============================================================================
    # EVALUATE MODEL PERFORMANCE
    # ============================================================================
    
    # Get chain ladder baseline for test set
    cl_dev_test = X_test[[f'cl_dev_{i}' for i in range(9)]].reset_index(drop=True)
    cl_dev_unscaled = pd.DataFrame(
        clscaler.inverse_transform(cl_dev_test),
        columns=[f'cumulative_paid_{i}' for i in range(9)],
        index=cl_dev_test.index
    )
    
    # Unscale actual residuals to get actual losses
    yu = unscaled_pred(y_test.reset_index(drop=True))
    yu.columns = [f'cumulative_paid_{i}' for i in range(9)]
    y_test_u = yu + cl_dev_unscaled  # Add back chain ladder to get actual losses
    
    # Get model predictions
    pred_scaled = model.predict(X_test.drop(columns=['year', 'cocode'] + 
                                           [f'cl_dev_{i}' for i in range(9)]))
    pu = unscaled_pred(pd.DataFrame(pred_scaled, columns=y_test.columns))
    pu.columns = yu.columns
    predictions = pu + cl_dev_unscaled  # Add chain ladder to residual predictions
    
    # Calculate mean absolute error
    ae = (predictions - y_test_u).abs()
    return model, {col: np.mean(ae[col]) for col in ae.columns}

def train_nn_extra(Xi, yi, base, old_cols, new_cols):
    """
    Train neural network using transfer learning.
    Starts with weights from base model and adds new features.
    
    Args:
        Xi: Feature dataframe
        yi: Target dataframe
        base: Pre-trained base model
        old_cols: Features used in base model
        new_cols: New features to add
    
    Returns:
        Dictionary of mean absolute errors by development period
    """
    # Select all features
    Xi = Xi[old_cols + new_cols + ['cocode','year'] + [f'cl_dev_{i}' for i in range(9)]]
    X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=0.2)
    
    input_dim = X_train.drop(columns=['cocode','year'] + 
                            [f'cl_dev_{i}' for i in range(9)]).shape[1]
    
    # Build model with same architecture
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(9))
    
    # ============================================================================
    # TRANSFER LEARNING: INITIALIZE WITH BASE MODEL WEIGHTS
    # ============================================================================
    
    base_weights = base.get_weights()
    old_weights, bias = base_weights[0], base_weights[1]
    n_old = old_weights.shape[0]
    n_new = input_dim - n_old
    
    if n_new > 0:
        # Add zero weights for new features
        new_weights = np.zeros((n_new, old_weights.shape[1]))
        mod_weights = np.vstack((old_weights, new_weights))
        base_weights[0] = mod_weights
    
    base_weights[1] = bias
    model.set_weights(base_weights)
    
    # Freeze first layer to preserve learned patterns
    model.layers[0].trainable = False
    
    # Compile and train
    model.compile(optimizer=Adam(learning_rate=1e-05, clipnorm=1.0), 
                 loss='mae', 
                 metrics=['mae'])
    model.fit(X_train.drop(columns=['cocode','year'] + [f'cl_dev_{i}' for i in range(9)]),
             y_train,
             epochs=100,
             batch_size=8,
             validation_split=0.2,
             callbacks=[early_stop])
    
    # Evaluate performance (same as train_nn)
    cl_dev_test = X_test[[f'cl_dev_{i}' for i in range(9)]].reset_index(drop=True)
    cl_dev_unscaled = pd.DataFrame(
        clscaler.inverse_transform(cl_dev_test),
        columns=[f'cumulative_paid_{i}' for i in range(9)],
        index=cl_dev_test.index
    )
    
    yu = unscaled_pred(y_test.reset_index(drop=True))
    yu.columns = [f'cumulative_paid_{i}' for i in range(9)]
    y_test_u = yu + cl_dev_unscaled
    
    pred_scaled = model.predict(X_test.drop(columns=['year', 'cocode'] + 
                                           [f'cl_dev_{i}' for i in range(9)]))
    pu = unscaled_pred(pd.DataFrame(pred_scaled, columns=y_test.columns))
    pu.columns = yu.columns
    predictions = pu + cl_dev_unscaled
    
    ae = (predictions - y_test_u).abs()
    print(ae.columns)
    return {col: np.mean(ae[col]) for col in ae.columns}

def run(iters, Xi, yi, old_cols, new_cols):
    """
    Run multiple iterations of model training to get robust results.
    
    Args:
        iters: Number of iterations
        Xi: Feature dataframe
        yi: Target dataframe
        old_cols: Base model features
        new_cols: Additional features for transfer learning
    
    Returns:
        base_dict: Results from base model
        x_dict: Results from transfer learning model
    """
    base_dict = {f'cumulative_paid_{i}': [] for i in range(9)}
    x_dict = {f'cumulative_paid_{i}': [] for i in range(9)}
    
    for i in range(iters + 1):
        # Train base model
        base_model, base_results = train_nn(Xi, yi, cols=old_cols)
        # Train transfer learning model
        x_results = train_nn_extra(Xi, yi, base_model, old_cols, new_cols)
        
        # Store results
        for j in range(9):
            base_dict[f'cumulative_paid_{j}'].append(base_results[f'cumulative_paid_{j}'])
            x_dict[f'cumulative_paid_{j}'].append(x_results[f'cumulative_paid_{j}'])
    
    return base_dict, x_dict

def print_results(dic):
    """Print mean and standard deviation of results"""
    for key in dic:
        print(f'{key}: mean: {np.mean(dic[key])} | sd: {np.std(dic[key])}')

# ============================================================================
# BASELINE: CHAIN LADDER PERFORMANCE
# ============================================================================

print("\n" + "="*60)
print("EVALUATING CHAIN LADDER BASELINE")
print("="*60)

# Load data with chain ladder predictions
X_cl = pd.read_csv('lag_data_scaled_0316.csv')
cl = X_cl[[f'cl_dev_{i}' for i in range(9)]]  # Chain ladder predictions
cl_answers = X_cl[[f'cumulative_paid_{i}' for i in range(9)]]  # Actual values
cl.columns = cl_answers.columns

# Calculate chain ladder errors
ae_cl = (cl - cl_answers).abs()
ae_cl = winsorize_df(ae_cl)

print("\nChain Ladder Mean Absolute Errors:")
for col in ae_cl.columns: 
    print(f'  {col}: {np.mean(ae_cl[col]):.2f}')

# ============================================================================
# COMPARE WITH ACTUAL COMPANY PREDICTIONS (IF AVAILABLE)
# ============================================================================

print("\n" + "="*60)
print("LOADING COMPANY PREDICTIONS (IF AVAILABLE)")
print("="*60)

try:
    # Load actual company predictions from saved dictionary
    X_ca = pd.read_csv('lag_data_0419.csv')
    with open('/Users/caleb/CL/AMDAOdictionary.pkl','rb') as file:
        dic = pickle.load(file)
    
    # Process company predictions
    all_list = []
    for key, value in dic.items():
        df = value[0].iloc[1:].reset_index(drop=True)
        ests = df['ABE+DAMD']
        yr, code = eval(key)
        
        if yr not in X_ca['year'].values or code not in X_ca['cocode'].values:
            continue
        
        ests = ests.iloc[::-1]  # Reverse order
        ests_vals = ests.iloc[:9].tolist()
        if len(ests_vals) < 9 or any(pd.isna(ests_vals)):
            continue
        
        ests_dict = {f'est_{i}': ests_vals[i] for i in range(9)}
        ests_dict.update({'yr': yr, 'code': code})
        all_list.append(ests_dict)
    
    # Combine and merge with actual values
    all = pd.DataFrame(all_list)
    X_ca_filtered = X_ca[['cocode', 'year'] + [f'cumulative_paid_{i}' for i in range(9)]]
    X_ca_merged = pd.merge(X_ca_filtered, all, 
                           left_on=['cocode', 'year'], 
                           right_on=['code', 'yr'], 
                           how='inner')
    
    # Calculate errors
    diff_df = X_ca_merged[['cocode', 'year']].copy()
    abs_diff_df = X_ca_merged[['cocode', 'year']].copy()
    
    for i in range(9):
        diff = X_ca_merged[f'cumulative_paid_{i}'] - X_ca_merged[f'est_{i}']
        diff_df[f'diff_{i}'] = diff
        abs_diff_df[f'abs_diff_{i}'] = diff.abs()
    
    abs_diff_df = winsorize_df(abs_diff_df, lower=0.01, upper=0.99)
    print(f"Loaded {len(abs_diff_df)} company predictions for comparison")
    
except Exception as e:
    print(f"Could not load company predictions: {e}")

# ============================================================================
# TRAIN MODELS WITH DIFFERENT FEATURE SETS
# ============================================================================

print("\n" + "="*60)
print("TRAINING NEURAL NETWORK MODELS")
print("="*60)

# Define feature sets
cl_data_cols = [f'dev_{i}' for i in range(9)] + \
               [f'X{i}.{i+12}' for i in range(12,120,12)]

print("\nTraining models with 100 iterations each...")
print("This may take several minutes per model...\n")

# Model 1: Chain ladder data + CL increase indicator
print("Training Model C1: Chain ladder features + CL increase indicator")
c1, c1_enhanced = run(100, X, y_s, cl_data_cols, ['CL_increase'])

# Model 2: Chain ladder data + full triangle + peer averages
print("Training Model M3: Chain ladder + full triangle + peer averages")
m3, m3_enhanced = run(100, X, y_s, cl_data_cols, 
                     [f'ft_{i}' for i in range(55)] + [f'avg_ft_{i}' for i in range(55)])

# Model 3: Chain ladder data + economic indicators
print("Training Model E1: Chain ladder + economic indicators")
e1, e1_enhanced = run(100, X, y_s, cl_data_cols, ['wPOP','wEMP','wINC'])

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Store all results in dictionary
peer_dicts = {
    'c1': c1,
    'c1_enhanced': c1_enhanced,
    'm3': m3,
    'm3_enhanced': m3_enhanced,
    'e1': e1,
    'e1_enhanced': e1_enhanced
}

# Save each model's results to CSV
for name, data in peer_dicts.items():
    df = pd.DataFrame(data)
    df.to_csv(f"{name}_us.csv", index=False)
    print(f"  Saved {name}_us.csv")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print("\nModels trained:")
print("  - C1: Chain ladder baseline with CL increase indicator")
print("  - M3: Full model with triangle features and peer averages")
print("  - E1: Model with economic indicators")
print("\nEach model has two versions:")
print("  - Base: Using only base features")
print("  - Enhanced: Using transfer learning with additional features")
print("\nResults saved as CSV files for further analysis")