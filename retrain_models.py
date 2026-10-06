#!/usr/bin/env python3
"""
Automated Model Retraining Script for Predictive Maintenance System.

This script replicates the complete training pipeline from final_project.ipynb
to retrain both Stage 1 (failure detection) and Stage 2 (failure type classification)
models using the current scikit-learn version installed in the environment.

This ensures the .pkl model files are compatible with the current environment,
avoiding "ValueError: node array from the pickle has an incompatible dtype" errors.

Usage:
    python3 retrain_models.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import sklearn


def treat_outliers(df, n=1.5, features=None):
    """
    Treat outliers using IQR method (matches notebook exactly).
    Clips values outside [Q1 - n*IQR, Q3 + n*IQR].
    """
    if features is None:
        features = []
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - n * IQR
        upper_bound = Q3 + n * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def load_and_preprocess_data(data_path):
    """
    Load the CSV data and perform initial preprocessing steps
    that match the notebook exactly.
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Drop UDI and Product ID columns (as done in the notebook)
    columns_to_drop = ["UDI", "Product ID"]
    existing_columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
        print(f"Dropped columns: {existing_columns_to_drop}")
    
    # Rename columns by replacing special characters (matches notebook)
    df.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_') for col in df.columns]
    print(f"Renamed columns: {list(df.columns)}")
    
    # Add engineered features (as done in the notebook)
    if "Process temperature _K_" in df.columns and "Air temperature _K_" in df.columns:
        df["temperature_difference"] = df["Process temperature _K_"] - df["Air temperature _K_"]
    
    if "Torque _Nm_" in df.columns and "Rotational speed _rpm_" in df.columns:
        df["Mechanical Power _W_"] = df["Torque _Nm_"] * df["Rotational speed _rpm_"] * (2 * np.pi / 60)
    
    print(f"Data shape after preprocessing: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    df = pd.get_dummies(df, columns=["Type"], dtype=float)
    return df


def train_stage1_model(df):
    """
    Train Stage 1 model (binary failure detection) exactly as the notebook does.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Binary Failure Detection")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop(columns=["Target", "Failure Type"])
    y = df["Target"]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data (matches notebook: test_size=0.2, random_state=42, stratify=y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling with StandardScaler (matches notebook)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Apply SMOTE for class balancing (matches notebook)
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)
    print(f"After SMOTE - Training set shape: {x_train_resampled.shape}")
    
    # Define models and parameter grids (matches notebook/model_utils.py)
    models = [
        ("CART", DecisionTreeClassifier(random_state=42)),
        ("RF", RandomForestClassifier(random_state=42)),
        ("XGB", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    ]
    
    param_grids = {
        "CART": {
            "max_depth": [3, 6, 9, 12],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [2, 4, 6],
            "max_features": [None, 2, 4]
        },
        "RF": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", "log2", None]
        },
        "XGB": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    }
    
    # GridSearchCV for each model (matches notebook)
    best_models = []
    best_scores = {}
    
    for model_name, model in models:
        print(f"\nTraining {model_name}...")
        param_grid = param_grids.get(model_name, {})
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1
        )
        
        grid_search.fit(x_train_resampled, y_train_resampled)
        best_models.append((model_name, grid_search.best_estimator_))
        best_scores[model_name] = grid_search.best_score_
        print(f"  Best CV Score: {grid_search.best_score_:.4f}")
        print(f"  Best Params: {grid_search.best_params_}")
    
    # Select top 3 models for VotingClassifier (matches notebook)
    top_3_models = sorted(best_scores, key=best_scores.get, reverse=True)[:3]
    print(f"\nTop 3 models: {top_3_models}")
    
    voting_clf = VotingClassifier(
        estimators=[(name, dict(best_models)[name]) for name in top_3_models],
        voting='soft'
    )
    
    voting_clf.fit(x_train_resampled, y_train_resampled)
    
    # Evaluate on test set
    y_pred = voting_clf.predict(x_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nStage 1 VotingClassifier Test Accuracy: {test_accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save the model
    model_path = "stage1_voting_classifier.pkl"
    joblib.dump(voting_clf, model_path)
    print(f"Stage 1 model saved to: {model_path}")
    
    return voting_clf, test_accuracy


def train_stage2_model(df):
    """
    Train Stage 2 model (failure type classification) exactly as the notebook does.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Failure Type Classification")
    print("=" * 60)
    
    # Filter to failure-only data (matches notebook)
    df_failure_only = df[df["Target"] == 1].copy()
    print(f"Failure-only data shape: {df_failure_only.shape}")
    
    # Separate features and target
    X2 = df_failure_only.drop(columns=["Target", "Failure Type"])
    y2 = df_failure_only["Failure Type"]
    
    print(f"Features shape: {X2.shape}")
    print(f"Failure type distribution:\n{y2.value_counts()}")
    
    # Encode target labels
    le2 = LabelEncoder()
    y2_encoded = le2.fit_transform(y2)
    print(f"Label classes: {list(le2.classes_)}")
    
    # Split data (matches notebook)
    try:
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2_encoded, test_size=0.2, random_state=42, stratify=y2_encoded
        )
    except ValueError:
        # If stratify fails due to small classes, fall back without stratify
        print("Warning: stratify failed, splitting without stratification")
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2_encoded, test_size=0.2, random_state=42
        )
    
    # Feature scaling with StandardScaler (matches notebook)
    scaler2 = StandardScaler()
    X2_train_scaled = scaler2.fit_transform(X2_train)
    X2_test_scaled = scaler2.transform(X2_test)
    
    # Apply SMOTE (matches notebook)
    try:
        smote = SMOTE(random_state=42)
        X2_train_resampled, y2_train_resampled = smote.fit_resample(X2_train_scaled, y2_train)
        print(f"After SMOTE - Training set shape: {X2_train_resampled.shape}")
    except ValueError:
        # If SMOTE fails (e.g., too few samples in a class), use original data
        print("Warning: SMOTE failed, using original training data")
        X2_train_resampled, y2_train_resampled = X2_train_scaled, y2_train
    
    # Define models and parameter grids (matches notebook/model_utils.py)
    models = [
        ("CART", DecisionTreeClassifier(random_state=42)),
        ("RF", RandomForestClassifier(random_state=42)),
        ("XGB", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    ]
    
    param_grids = {
        "CART": {
            "max_depth": [3, 6, 9, 12],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [2, 4, 6],
            "max_features": [None, 2, 4]
        },
        "RF": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", "log2", None]
        },
        "XGB": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    }
    
    # GridSearchCV for each model (matches notebook)
    best_models = []
    best_scores = {}
    
    for model_name, model in models:
        print(f"\nTraining {model_name}...")
        param_grid = param_grids.get(model_name, {})
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1
        )
        
        grid_search.fit(X2_train_resampled, y2_train_resampled)
        best_models.append((model_name, grid_search.best_estimator_))
        best_scores[model_name] = grid_search.best_score_
        print(f"  Best CV Score: {grid_search.best_score_:.4f}")
        print(f"  Best Params: {grid_search.best_params_}")
    
    # Select top 3 models for VotingClassifier (matches notebook)
    top_3_models = sorted(best_scores, key=best_scores.get, reverse=True)[:3]
    print(f"\nTop 3 models: {top_3_models}")
    
    voting_clf2 = VotingClassifier(
        estimators=[(name, dict(best_models)[name]) for name in top_3_models],
        voting='soft'
    )
    
    voting_clf2.fit(X2_train_resampled, y2_train_resampled)
    
    # Evaluate on test set
    y2_pred = voting_clf2.predict(X2_test_scaled)
    test_accuracy = accuracy_score(y2_test, y2_pred)
    print(f"\nStage 2 VotingClassifier Test Accuracy: {test_accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y2_test, y2_pred, target_names=le2.classes_)}")
    
    # Save the model
    model_path = "stage2_voting_classifier.pkl"
    joblib.dump(voting_clf2, model_path)
    print(f"Stage 2 model saved to: {model_path}")
    
    # Save the label encoder for Stage 2 (needed for decoding predictions)
    le_path = "stage2_label_encoder.pkl"
    joblib.dump(le2, le_path)
    print(f"Stage 2 LabelEncoder saved to: {le_path}")
    
    return voting_clf2, le2, test_accuracy


def verify_models():
    """
    Verify that the saved models can be loaded successfully.
    """
    print("\n" + "=" * 60)
    print("VERIFICATION: Loading saved models")
    print("=" * 60)
    
    success = True
    
    for model_file in ["stage1_voting_classifier.pkl", "stage2_voting_classifier.pkl", "stage2_label_encoder.pkl"]:
        try:
            model = joblib.load(model_file)
            print(f"  ✓ Successfully loaded: {model_file} ({type(model).__name__})")
        except Exception as e:
            print(f"  ✗ Failed to load {model_file}: {e}")
            success = False
    
    return success


def main():
    """
    Main entry point for the retraining script.
    """
    print("=" * 60)
    print("Predictive Maintenance - Model Retraining Script")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"Working directory: {os.getcwd()}")
    
    # Determine the data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "predictive_maintenance.csv")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        sys.exit(1)
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(data_path)
    
    # Step 2: Train Stage 1 model (binary failure detection)
    stage1_model, stage1_accuracy = train_stage1_model(df)
    
    # Step 3: Train Stage 2 model (failure type classification)
    stage2_model, stage2_le, stage2_accuracy = train_stage2_model(df)
    
    # Step 4: Verify models can be loaded
    verification_passed = verify_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Stage 1 Test Accuracy: {stage1_accuracy:.4f}")
    print(f"Stage 2 Test Accuracy: {stage2_accuracy:.4f}")
    print(f"Model Verification: {'PASSED' if verification_passed else 'FAILED'}")
    print(f"scikit-learn version used: {sklearn.__version__}")
    print("\nFiles generated:")
    print("  - stage1_voting_classifier.pkl")
    print("  - stage2_voting_classifier.pkl")
    print("  - stage2_label_encoder.pkl")
    
    if verification_passed:
        print("\n✓ Retraining completed successfully!")
        print("  The Streamlit app should now be able to load the models without errors.")
    else:
        print("\n✗ Retraining completed but verification failed.")
        print("  Please check the error messages above.")
    
    return 0 if verification_passed else 1


if __name__ == "__main__":
    sys.exit(main())
