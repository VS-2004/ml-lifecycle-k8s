import pandas as pd
from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=5000,n_features=10,n_informative=6,n_redundant=2,random_state=42)
cols = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df["target"] = y
df.to_csv("data/dataset.csv", index=False)
print(f"Dataset saved — {df.shape[0]} rows, {df.shape[1]} columns")
