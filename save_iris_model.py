import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris dataset
data = pd.read_csv("Iris dataset.csv")

# 2. Select the input features
X = data[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

# 3. Select the target
y = data["species"]

# 4. Create the Logistic Regression pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. Train using the full dataset
model.fit(X, y)

# 6. Save the trained model
joblib.dump(model, "iris_model.pkl")

print("Iris model successfully trained and saved!")
print("Model file: iris_model.pkl")
print("Flower species:", list(model.classes_))