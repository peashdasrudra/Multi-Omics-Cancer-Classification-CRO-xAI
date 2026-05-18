import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path="data/brca_data_w_subtypes.csv"):
    """Loads the BRCA multi-omics dataset from the specified CSV path."""
    df = pd.read_csv(path)
    print(f"Raw dataset shape: {df.shape}")
    return df

def deduplicate_columns(df):
    """
    Removes columns with identical content (not just identical names).
    Pandas auto-renames duplicate column names on CSV load (.1, .2 suffixes),
    but 99 cn_ columns have identical values because they represent genes at
    the same chromosomal locus with identical copy-number profiles.
    This content-based dedup reduces features from 1,936 → 1,837.
    """
    shape_before = df.shape[1]
    df_dedup = df.T.drop_duplicates().T
    removed = shape_before - df_dedup.shape[1]
    print(f"Removed {removed} content-duplicate columns. Shape: {df_dedup.shape}")
    return df_dedup

def inspect_targets(df):
    """Prints the distribution of classes for the clinical targets."""
    targets = ["ER.Status", "HER2.Final.Status", "histological.type"]
    for t in targets:
        print(f"\n{t} distribution:")
        print(df[t].value_counts(dropna=False))

def save_cleaned_data(df, path="outputs/preprocessed/cleaned_data.csv"):
    """Saves the deduplicated dataset to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to {path}")

def separate_features_targets(df, target_name):
    """
    Separates the features (X) from the specific target (y).
    Automatically drops rows where the target value is missing (NaN).
    """
    # Drop rows where the target is NaN
    df_clean = df.dropna(subset=[target_name]).copy()
    
    # Exclude all clinical target columns from the feature set
    clinical_cols = ["ER.Status", "HER2.Final.Status", "histological.type", "vital.status", "PR.Status"]
    feature_cols = [c for c in df_clean.columns if c not in clinical_cols]
    
    X = df_clean[feature_cols].copy()
    y = df_clean[target_name].copy()
    return X, y

def encode_target(y, target_name):
    """
    Encodes categorical targets into numeric values.
    Maps Positive/Negative to 1/0 for binary targets.
    Uses LabelEncoder for multi-class targets.
    """
    if target_name in ["ER.Status", "HER2.Final.Status", "PR.Status"]:
        # Direct mapping for binary clinical status
        y_mapped = y.map({"Positive": 1, "Negative": 0})
        return y_mapped
    else:
        # Label encoding for multi-class or text targets (e.g., histological type)
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y), name=y.name, index=y.index)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    Uses 'stratify=y' to preserve class proportions in both sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scales features to have 0 mean and 1 variance using StandardScaler.
    Fit is performed ONLY on the training set to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def apply_smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE to oversample the minority class in the training data.
    This is used for the final model training and future CRO steps.
    """
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def save_feature_names(feature_names, target_name, out_dir="outputs/preprocessed"):
    """Saves feature column names as a numpy array for SHAP plot labels."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/{target_name}_feature_names.npy", np.array(feature_names))
    print(f"Saved {len(feature_names)} feature names for {target_name}")

def save_processed(X_train, X_test, y_train, y_test, target_name, out_dir="outputs/preprocessed"):
    """Saves the processed numpy arrays for future use (like DC-CRO)."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/{target_name}_X_train.npy", X_train)
    np.save(f"{out_dir}/{target_name}_X_test.npy", X_test)
    np.save(f"{out_dir}/{target_name}_y_train.npy", y_train)
    np.save(f"{out_dir}/{target_name}_y_test.npy", y_test)
    print(f"Saved processed .npy files for {target_name} to {out_dir}")
