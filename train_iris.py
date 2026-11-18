import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. LOAD THE DATASET
data = pd.read_csv("Iris dataset.csv")

# 2. SEPARATE INPUT FEATURES AND TARGET
X = data[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]

y = data["species"]

# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("TRAINING DATA SIZE:", len(X_train))
print("TESTING DATA SIZE:", len(X_test))


# 4. CREATE THE THREE CLASSIFIERS
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
}


# 5. TRAIN AND EVALUATE EACH MODEL
results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name.upper())
    print("=" * 50)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, average="macro"
    )

    recall = recall_score(
        y_test, predictions, average="macro"
    )

    f1 = f1_score(
        y_test, predictions, average="macro"
    )

    # Display results
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions))

    # Save results for comparison
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # Create and save confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display.plot(cmap="Blues")
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(filename)
    plt.show()


# 6. COMPARE THE THREE MODELS
comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

# Save comparison results
comparison.to_csv("model_comparison.csv", index=False)

print("\nTraining and evaluation completed!")
print("Results saved in your project folder.")