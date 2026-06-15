import mlflow.sklearn
import os
import glob

def load_model():
    base = os.getenv("MODEL_PATH", "/app/mlruns")
    patterns = [
        os.path.join(base, "*/models/*/artifacts/MLmodel"),
        os.path.join(base, "*/*/artifacts/model/MLmodel"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No MLmodel found under {base}")
    model_dir = os.path.dirname(sorted(matches)[-1])
    model = mlflow.sklearn.load_model(model_dir)
    print(f"Model loaded from: {model_dir}")
    return model