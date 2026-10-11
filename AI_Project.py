import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

if not os.path.exists("products.csv"):
    rng = np.random.default_rng(42)
    sample = pd.DataFrame(
        rng.uniform(0, 10, size=(200, 3)),
        columns=["feature1", "feature2", "feature3"],
    )
    sample["label"] = (sample.sum(axis=1) // 10).astype(int)
    sample.to_csv("products.csv", index=False)

data = pd.read_csv("products.csv")
features = data[["feature1", "feature2", "feature3"]]
labels = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

def recommend(product_features):
    row = pd.DataFrame([product_features], columns=features.columns)
    return model.predict(row)[0]

example_product = [1.0, 2.0, 3.0]
print(f"Recommended Product: {recommend(example_product)}")