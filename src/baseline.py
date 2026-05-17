import numpy as np
import pandas as pd
import os
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone

def evaluate_cv(model, X, y, cv=5, random_state=42):
    """
    Evaluates a model using Stratified 5-Fold Cross-Validation.
    Crucially uses ImbPipeline to apply SMOTE *only* on the training folds
    to prevent data leakage and ensure realistic evaluation metrics.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Dictionary to store scores for each fold
    scores = {"accuracy": [], "f1_weighted": [], "f1_macro": [], 
              "precision": [], "recall": [], "roc_auc": [], "mcc": []}
    
    # Pipeline ensures SMOTE is fit ONLY on training folds during CV
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', model)
    ])
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Clone the pipeline to ensure a fresh model for each fold
        pipe_clone = clone(pipeline)
        pipe_clone.fit(X_tr, y_tr)
        
        # Predict on validation fold
        y_pred = pipe_clone.predict(X_val)
        
        # Calculate standard classification metrics
        scores["accuracy"].append(accuracy_score(y_val, y_pred))
        scores["f1_weighted"].append(f1_score(y_val, y_pred, average="weighted"))
        scores["f1_macro"].append(f1_score(y_val, y_pred, average="macro"))
        scores["precision"].append(precision_score(y_val, y_pred, average="weighted", zero_division=0))
        scores["recall"].append(recall_score(y_val, y_pred, average="weighted"))
        
        # Calculate ROC AUC (differs for binary vs multi-class)
        if len(np.unique(y)) == 2:
            y_proba = pipe_clone.predict_proba(X_val)[:, 1]
            scores["roc_auc"].append(roc_auc_score(y_val, y_proba))
        else:
            y_proba = pipe_clone.predict_proba(X_val)
            scores["roc_auc"].append(roc_auc_score(y_val, y_proba, multi_class="ovr"))
        
        scores["mcc"].append(matthews_corrcoef(y_val, y_pred))
    
    # Return the mean and standard deviation for each metric
    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}

def run_baselines(X_train, y_train, target_name):
    """
    Runs the baseline evaluation for Random Forest, SVM, and XGBoost
    on the provided training data.
    """
    models = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=42),
        "XGB": XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss")
    }
    
    results = {}
    for name, model in models.items():
        print(f"Running {name} on {target_name}...")
        results[name] = evaluate_cv(model, X_train, y_train)
        
    return results

def results_to_dataframe(results_dict):
    """Formats the results dictionary into a clean pivoted Pandas DataFrame."""
    rows = []
    for model, metrics in results_dict.items():
        for metric, (mean, std) in metrics.items():
            rows.append([model, metric, f"{mean:.4f} ± {std:.4f}"])
    df = pd.DataFrame(rows, columns=["Model", "Metric", "Mean ± Std"])
    
    # Pivot to make it professional (Models as rows, Metrics as columns)
    df_pivot = df.pivot(index="Model", columns="Metric", values="Mean ± Std")
    
    # Reorder columns logically
    cols = ["accuracy", "f1_weighted", "f1_macro", "precision", "recall", "roc_auc", "mcc"]
    cols = [c for c in cols if c in df_pivot.columns]
    df_pivot = df_pivot[cols]
    
    # Reset index to keep 'Model' as a column for easy CSV saving
    return df_pivot.reset_index()

