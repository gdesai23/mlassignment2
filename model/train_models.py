from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "model_metrics.csv"

RANDOM_STATE = 42
TARGET_COLUMN = "income"


def load_adult_dataset() -> pd.DataFrame:
    """Load the Adult dataset from cached local files or UCI source."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_cache = DATA_DIR / "adult_train.csv"
    test_cache = DATA_DIR / "adult_test.csv"

    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        TARGET_COLUMN,
    ]

    if train_cache.exists() and test_cache.exists():
        train_df = pd.read_csv(train_cache)
        test_df = pd.read_csv(test_cache)
    else:
        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

        train_df = pd.read_csv(
            train_url,
            names=column_names,
            header=None,
            skipinitialspace=True,
        )
        test_df = pd.read_csv(
            test_url,
            names=column_names,
            header=None,
            skiprows=1,
            comment="|",
            skipinitialspace=True,
        )

        train_df.to_csv(train_cache, index=False)
        test_df.to_csv(test_cache, index=False)

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    full_df[TARGET_COLUMN] = full_df[TARGET_COLUMN].str.replace(".", "", regex=False).str.strip()

    for col in full_df.columns:
        if full_df[col].dtype == object:
            full_df[col] = full_df[col].str.strip()

    full_df.replace("?", np.nan, inplace=True)
    full_df.dropna(inplace=True)

    full_df[TARGET_COLUMN] = full_df[TARGET_COLUMN].map({"<=50K": 0, ">50K": 1})

    return full_df


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocessor(feature_frame: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = feature_frame.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = feature_frame.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numerical_cols, categorical_cols


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "kNN": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=11)),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", GaussianNB()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=16,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=350,
                        learning_rate=0.08,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_adult_dataset()
    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_export = x_test.copy()
    test_export[TARGET_COLUMN] = y_test.values
    test_export.to_csv(TEST_DATA_PATH, index=False)

    preprocessor, _, _ = create_preprocessor(x_train)
    models = build_models(preprocessor)

    results = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        metrics, cm = evaluate_model(model, x_test, y_test)

        model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path, compress=3)

        cm_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
        pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["Actual_0", "Actual_1"]).to_csv(cm_path)

        results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    results_df.to_csv(METRICS_PATH, index=False)

    print("Training completed successfully.")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved test dataset to: {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
