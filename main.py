import pandas as pd
from data_preprocessing import preprocess_stage1, preprocess_stage2
from model_utils import train_stage1, train_stage2
from evaluation import evaluate_model
import joblib

df = pd.read_csv("predictive_maintenance.csv")

x_train, x_test, y_train, y_test, le = preprocess_stage1(df)
voting_clf, best_scores = train_stage1(x_train, y_train)
print("\nStage 1 Model Evaluation:")
evaluate_model(voting_clf, x_test, y_test, le)

df_failure_only = df[df["Target"] == 1]
X2_train, X2_test, y2_train, y2_test, le2 = preprocess_stage2(df_failure_only)
voting_clf2, best_scores2 = train_stage2(X2_train, y2_train)
print("\nStage 2 Model Evaluation (Failure Type Prediction):")
evaluate_model(voting_clf2, X2_test, y2_test, le2)

print("\nModels trained and saved successfully.")