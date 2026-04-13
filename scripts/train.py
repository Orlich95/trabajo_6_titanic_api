import io
import json
import os
from datetime import datetime

import boto3
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mlops-tp4-orlich-2026")
DATA_S3_KEY = os.getenv("DATA_S3_KEY", "tp7/data/titanic.csv")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY", "tp7/models/titanic_model.joblib")
METRICS_S3_KEY = os.getenv("METRICS_S3_KEY", "tp7/results/metrics.json")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "models/titanic_model.joblib")


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def load_dataset_from_s3():
    s3 = get_s3_client()
    response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=DATA_S3_KEY)
    return pd.read_csv(io.BytesIO(response["Body"].read()))


def build_pipeline():
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return model


def ensure_model_dir():
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)


def main():
    df = load_dataset_from_s3()

    expected_columns = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas en el dataset: {missing_columns}")

    df = df[expected_columns].copy()

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    ensure_model_dir()
    joblib.dump(pipeline, MODEL_LOCAL_PATH)

    s3 = get_s3_client()
    s3.upload_file(MODEL_LOCAL_PATH, S3_BUCKET_NAME, MODEL_S3_KEY)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "model_s3_key": MODEL_S3_KEY,
        "data_s3_key": DATA_S3_KEY,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=METRICS_S3_KEY,
        Body=json.dumps(metrics, indent=2),
        ContentType="application/json",
    )

    print("Reentrenamiento completado.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()