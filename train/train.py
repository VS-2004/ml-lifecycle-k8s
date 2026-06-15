import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
df = pd.read_csv("data/dataset.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
mlflow.set_experiment("ml-lifecycle-k8s")
with mlflow.start_run(run_name="random-forest-baseline"):
    params = {"n_estimators": 100, "max_depth": 8, "min_samples_split": 5, "random_state": 42}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="ChurnClassifier")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(classification_report(y_test, y_pred))
    print("Model registered in MLflow as: ChurnClassifier")
