import os
import pandas as pd
from src.utils import set_seed
from src.preprocess import (load_data, deduplicate_columns, inspect_targets, save_cleaned_data,
                            separate_features_targets, encode_target, split_data, scale_features, 
                            apply_smote, save_processed, save_feature_names)
from src.baseline import run_baselines, results_to_dataframe

def main():
    # 1. Ensure reproducibility
    set_seed(42)
    
    # Create output directories if they don't exist
    os.makedirs("outputs/baseline_metrics", exist_ok=True)
    os.makedirs("outputs/preprocessed", exist_ok=True)
    
    print("=== Day 1: Data Loading & Deduplication ===")
    df = load_data()
    df = deduplicate_columns(df)
    inspect_targets(df)
    save_cleaned_data(df)
    
    # The 3 clinical targets requested in the thesis guide
    targets = ["ER.Status", "HER2.Final.Status", "histological.type"]
    
    for target in targets:
        print(f"\n=== Processing Target: {target} ===")
        
        # Separate features and the current target
        X, y = separate_features_targets(df, target)
        
        # Save feature names for SHAP explainability plots later
        save_feature_names(X.columns.tolist(), target)
        
        # Encode target labels to numbers
        y_enc = encode_target(y, target)
        
        # Filter out any remaining NaN targets (e.g. Indeterminate, Equivocal if mapped to NaN)
        valid_idx = y_enc.notna()
        X = X[valid_idx]
        y_enc = y_enc[valid_idx].astype(int)
        
        print(f"Shape after dropping target NaNs: X={X.shape}, y={y_enc.shape}")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = split_data(X, y_enc)
        
        # Scale features (fit on train, transform both)
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
        
        print("Saving un-resampled data (for proper CV pipeline execution)...")
        # Save pre-SMOTE scaled data so CV can run correctly without leakage
        save_processed(X_train_sc, X_test_sc, y_train.values, y_test.values, f"{target}_unresampled")
        
        print("Applying SMOTE on full training set (for final model and future CRO use)...")
        X_train_res, y_train_res = apply_smote(X_train_sc, y_train.values)
        save_processed(X_train_res, X_test_sc, y_train_res, y_test.values, target)
        
        print(f"=== Baseline Evaluation for {target} ===")
        # Evaluate using ImbPipeline on un-resampled data to avoid leakage during CV
        results = run_baselines(X_train_sc, y_train.values, target)
        
        # Convert results to dataframe and save to CSV
        df_res = results_to_dataframe(results)
        df_res.to_csv(f"outputs/baseline_metrics/{target}_baseline.csv", index=False)
        
        print(df_res)
        print(f"Done for {target}")

if __name__ == "__main__":
    main()
