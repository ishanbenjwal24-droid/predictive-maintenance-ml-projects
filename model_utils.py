import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

def train_stage1(x_train, y_train):
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

    best_models = []
    best_scores = {}

    for model_name, model in models:
        param_grid = param_grids.get(model_name, {})

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1
        )

        grid_search.fit(x_train, y_train)
        best_models.append((model_name, grid_search.best_estimator_))
        best_scores[model_name] = grid_search.best_score_

    top_3_models = sorted(best_scores, key=best_scores.get, reverse=True)[:3]
    voting_clf = VotingClassifier(
        estimators=[(name, dict(best_models)[name]) for name in top_3_models],
        voting='soft'
    )

    voting_clf.fit(x_train, y_train)
    joblib.dump(voting_clf, "stage1_voting_classifier.pkl")
    return voting_clf, best_scores

def train_stage2(X2_train_res, y2_train_res):
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

    best_models = []
    best_scores = {}

    for model_name, model in models:
        param_grid = param_grids.get(model_name, {})

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1
        )

        grid_search.fit(X2_train_res, y2_train_res)
        best_models.append((model_name, grid_search.best_estimator_))
        best_scores[model_name] = grid_search.best_score_

    top_3_models = sorted(best_scores, key=best_scores.get, reverse=True)[:3]
    voting_clf2 = VotingClassifier(
        estimators=[(name, dict(best_models)[name]) for name in top_3_models],
        voting='soft'
    )

    voting_clf2.fit(X2_train_res, y2_train_res)
    joblib.dump(voting_clf2, "stage2_voting_classifier.pkl")
    return voting_clf2, best_scores


