import mlflow.sklearn
import mlflow
import os
MODEL_NAME = "ChurnClassifier"
MODEL_STAGE = "models:/ChurnClassifier/latest"
def load_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model = mlflow.sklearn.load_model(MODEL_STAGE)
    print(f"Model loaded: {MODEL_STAGE}")
    return model
