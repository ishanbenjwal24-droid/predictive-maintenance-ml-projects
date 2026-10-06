import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_stage1(df):
    X = df.drop(columns=["Target", "Failure Type"])
    y = df["Target"]

    categorical_features = ["Type"]
    numerical_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    X_processed = preprocessor.fit_transform(X)

    if len(X_processed) > 1:
        x_train, x_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        return x_train, x_test, y_train_enc, y_test_enc, le
    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return X_processed, None, y_enc, None, le

def preprocess_stage2(df_failure_only):
    X2 = df_failure_only.drop(columns=["Target", "Failure Type"])
    y2 = df_failure_only["Failure Type"]

    categorical_features = ["Type"]
    numerical_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    X2_processed = preprocessor.fit_transform(X2)

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2_processed, y2, test_size=0.2, random_state=42, stratify=y2
    )

    le2 = LabelEncoder()
    y2_train_enc = le2.fit_transform(y2_train)
    y2_test_enc = le2.transform(y2_test)

    smote = SMOTE(random_state=42)
    X2_train_res, y2_train_res = smote.fit_resample(X2_train, y2_train_enc)

    return X2_train_res, X2_test, y2_train_res, y2_test_enc, le2


