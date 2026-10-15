
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib
import os

# 1. Load Dataset
df = pd.read_csv("aiml_product_reviews_corrected.csv")
os.makedirs("outputs", exist_ok=True)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())


# 2. Data Inspection
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# 3. Check Number of Sentiment Classes
print("\nNumber of Classes:")
print(df["sentiment"].nunique())

print("\nClasses:")
print(df["sentiment"].unique())

print("\nClass Distribution:")
print(df["sentiment"].value_counts())


# 4. Data Cleaning and Preprocessing

# Remove duplicate review IDs
df = df.drop_duplicates(subset=["review_id"])

# Fill missing text and category values
df["review_text"] = df["review_text"].fillna("").astype(str)
df["category"] = df["category"].fillna("Unknown").astype(str)

# Standardize sentiment labels
df["sentiment"] = (
    df["sentiment"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

# Remove rows with empty review text or invalid sentiment labels
valid_sentiments = ["negative", "neutral", "positive"]

df = df[
    (df["review_text"].str.strip() != "")
    & (df["sentiment"].isin(valid_sentiments))
].copy()

# Convert date column into datetime format
df["review_date"] = pd.to_datetime(
    df["review_date"], errors="coerce"
)

# Convert numeric columns to appropriate types
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["helpful_votes"] = pd.to_numeric(
    df["helpful_votes"], errors="coerce"
)
df["review_length_chars"] = pd.to_numeric(
    df["review_length_chars"], errors="coerce"
)

print("\nDataset after cleaning:")
print(df.head())

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 4.1 Label Encoding for Categorical Columns

from sklearn.preprocessing import LabelEncoder

# Create separate encoders
sentiment_encoder = LabelEncoder()
category_encoder = LabelEncoder()

# Encode sentiment labels
df["sentiment_encoded"] = sentiment_encoder.fit_transform(
    df["sentiment"]
)

# Encode product categories
df["category_encoded"] = category_encoder.fit_transform(
    df["category"]
)

# Display the encoding mappings
print("\nSentiment Label Encoding:")
for label, encoded_value in zip(
    sentiment_encoder.classes_,
    sentiment_encoder.transform(sentiment_encoder.classes_)
):
    print(f"{label} = {encoded_value}")

print("\nProduct Category Label Encoding:")
for label, encoded_value in zip(
    category_encoder.classes_,
    category_encoder.transform(category_encoder.classes_)
):
    print(f"{label} = {encoded_value}")

# Display the encoded dataset
print("\nEncoded Dataset:")
print(
    df[
        ["review_text", "sentiment", "sentiment_encoded",
         "category", "category_encoded"]
    ].head()
)

# Save the encoded dataset
df.to_csv("outputs/encoded_product_reviews.csv", index=False)

print("\nEncoded dataset saved successfully!")

# 5. Data Visualization - Sentiment Distribution

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="sentiment",
    order=["negative", "neutral", "positive"]
)

plt.title("Sentiment Distribution of Product Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()


# 6. Visualize Sentiment by Product Category

category_sentiment = pd.crosstab(
    df["category"],
    df["sentiment"]
)

category_sentiment.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("Sentiment by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 7. Review Rating Distribution

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="rating",
    order=sorted(df["rating"].dropna().unique())
)

plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()


# 8. Prepare the dataset
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("outputs", exist_ok=True)

# Remove rows with missing/invalid data
df = df.dropna(subset=["review_text", "sentiment"]).copy()
df["review_text"] = df["review_text"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

df = df[
    df["review_text"].ne("")
    & df["sentiment"].isin(["negative", "neutral", "positive"])
].copy()

# Remove the audit-only column; do not use it as a model feature
df = df.drop(columns=["original_sentiment"], errors="ignore")

X = df["review_text"]
y = df["sentiment"]

# 9. Group similar generated templates together.
# This removes the product name and review reference suffix
# when identifying the underlying review template.
def normalize_review(text):
    text = text.lower().strip()
    text = re.sub(
        r"\s*i purchased this .*?\.?\s*review reference \d+\.?$",
        "",
        text
    )
    return re.sub(r"\s+", " ", text).strip()

groups = X.apply(normalize_review)

print("Total rows:", len(df))
print("Unique review texts:", X.nunique())
print("Unique review templates:", groups.nunique())

# 10. Split by template, not by individual row
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print(
    "Overlapping templates:",
    len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
)

print("\nTraining sentiment distribution:")
print(y_train.value_counts())

print("\nTesting sentiment distribution:")
print(y_test.value_counts())

# 11. Define models using pipelines.
# TF-IDF is fitted only on the training data.
models = {
    "Decision Tree": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("classifier", DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# 12. Train and evaluate all models on the same test set
results = []
trained_models = {}

sentiment_labels = ["negative", "neutral", "positive"]

for name, model in models.items():
    print(f"\n{'=' * 15} {name} {'=' * 15}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=sentiment_labels,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=sentiment_labels,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=sentiment_labels
    )
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sentiment_labels,
        yticklabels=sentiment_labels
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    plt.savefig(
        f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.show()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    trained_models[name] = model

    # Save predictions for each model
    pd.DataFrame({
        "review_id": df.iloc[test_idx]["review_id"].values,
        "review_text": X_test.values,
        "actual_sentiment": y_test.values,
        "predicted_sentiment": predictions
    }).to_csv(
        f"outputs/{name.lower().replace(' ', '_')}_predictions.csv",
        index=False
    )

# 13. Compare models
comparison = pd.DataFrame(results).sort_values(
    by="Macro F1",
    ascending=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv("outputs/model_comparison.csv", index=False)

# Select by macro F1 so all three classes matter
best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(
    {
        "pipeline": best_model,
        "model_name": best_name,
        "sentiment_labels": sentiment_labels
    },
    "outputs/best_sentiment_model.joblib"
)

print(f"\nBest model by macro F1: {best_name}")
print(f"Accuracy: {comparison.iloc[0]['Accuracy'] * 100:.2f}%")
print(f"Macro F1: {comparison.iloc[0]['Macro F1']:.4f}")
print("\nAll models and results saved in the outputs folder.")



# Decision Tree Predictions and Evaluation

decision_tree = trained_models["Decision Tree"]

y_pred_tree = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:")
print(f"{tree_accuracy * 100:.2f}%")

sentiment_labels = ["negative", "neutral", "positive"]

print("\nDecision Tree Classification Report:")
print(classification_report(
    y_test,
    y_pred_tree,
    labels=sentiment_labels,
    zero_division=0
))

tree_cm = confusion_matrix(
    y_test,
    y_pred_tree,
    labels=sentiment_labels
)

print("\nDecision Tree Confusion Matrix:")
print(tree_cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    tree_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=sentiment_labels,
    yticklabels=sentiment_labels
)
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.show()



# 15. Decision Tree Evaluation

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:")
print(f"{tree_accuracy * 100:.2f}%")


# 16. Decision Tree Confusion Matrix

sentiment_labels = ["negative", "neutral", "positive"]

tree_cm = confusion_matrix(
    y_test,
    y_pred_tree,
    labels=sentiment_labels
)

print("\nDecision Tree Confusion Matrix:")
print(tree_cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    tree_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=sentiment_labels,
    yticklabels=sentiment_labels
)

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.show()



# Random Forest Predictions and Evaluation

random_forest = trained_models["Random Forest"]

y_pred_forest = random_forest.predict(X_test)

print("\nRandom Forest Predictions:")
print(y_pred_forest)

forest_accuracy = accuracy_score(y_test, y_pred_forest)

print("\nRandom Forest Accuracy:")
print(f"{forest_accuracy * 100:.2f}%")

sentiment_labels = ["negative", "neutral", "positive"]

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    y_pred_forest,
    labels=sentiment_labels,
    zero_division=0
))

forest_cm = confusion_matrix(
    y_test,
    y_pred_forest,
    labels=sentiment_labels
)

print("\nRandom Forest Confusion Matrix:")
print(forest_cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    forest_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=sentiment_labels,
    yticklabels=sentiment_labels
)
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.show()


# 21. Train Logistic Regression Baseline

tfidf_logistic = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_tfidf = tfidf_logistic.fit_transform(X_train)
X_test_tfidf = tfidf_logistic.transform(X_test)

# Train the model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create a complete text-classification pipeline
logistic_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train the model
logistic_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_logistic = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

print("Logistic Regression Accuracy:",
      round(logistic_accuracy * 100, 2), "%")

print(classification_report(
    y_test,
    y_pred_logistic,
    zero_division=0
))

# Classification Report
print("\nLogistic Regression Classification Report:")

print(classification_report(
    y_test,
    y_pred_logistic,
    labels=["negative", "neutral", "positive"],
    zero_division=0
))

# 22. Compare Models

print("\n MODEL COMPARISON ")

print(f"Decision Tree Accuracy: {tree_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {forest_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%")

model_comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression"
    ],
    "Accuracy": [
        tree_accuracy,
        forest_accuracy,
        logistic_accuracy
    ]
})

print("\nModel Comparison Table:")
print(model_comparison)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=model_comparison,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# 23. Save the Best Model

# Select the model with the highest measured test accuracy.
models = {
    "Decision Tree": (decision_tree, tree_accuracy),
    "Random Forest": (random_forest, forest_accuracy),
    "Logistic Regression": (logistic_model, logistic_accuracy)
}

best_model_name = max(
    models,
    key=lambda name: models[name][1]
)

best_model = models[best_model_name][0]
best_accuracy = models[best_model_name][1]

# Save both vectorizer and classifier.

# Save the best model pipeline
joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "sentiment_labels": sentiment_labels
    },
    "outputs/best_sentiment_model.joblib"
)

# Save model comparison results
model_comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("Best model and comparison results saved successfully.")

# Save predictions for reviewing errors.
predictions_df = pd.DataFrame({
    "review_id": df.loc[y_test.index, "review_id"].values,
    "actual_sentiment": y_test.values,
    "predicted_sentiment": best_model.predict(X_test)
})

predictions_df.to_csv(
    "outputs/test_predictions.csv",
    index=False
)

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy * 100:.2f}%")
print("\nModel and results saved in the outputs folder.")

# 24. Check for Overlapping Review Texts in Train and Test Sets
from sklearn.model_selection import GroupShuffleSplit

X = df["review_text"]
y = df["sentiment"]
groups = df["review_text"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(
    "Overlapping review texts:",
    len(set(X_train) & set(X_test))
)

# 25. Check for Unique Review Texts in Train and Test Sets
print("\nUnique training texts:", X_train.nunique())
print("Unique testing texts:", X_test.nunique())

print("\nTraining sentiment distribution:")
print(y_train.value_counts())

print("\nTesting sentiment distribution:")
print(y_test.value_counts())

print("\nActual test labels:")
print(y_test.value_counts().sort_index())


# 14. Final Week 2 Evaluation and Model Versioning

import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

os.makedirs("outputs", exist_ok=True)

sentiment_labels = ["negative", "neutral", "positive"]

# Verify that all three classes exist in both splits
print("\nTraining classes:")
print(sorted(y_train.unique()))

print("\nTesting classes:")
print(sorted(y_test.unique()))

if not set(sentiment_labels).issubset(set(y_train.unique())):
    raise ValueError(
        "Training split is missing a sentiment class. "
        "Review the grouped train/test split."
    )

if not set(sentiment_labels).issubset(set(y_test.unique())):
    raise ValueError(
        "Test split is missing a sentiment class. "
        "Review the grouped train/test split."
    )

# Select the best model using the existing comparison
comparison = comparison.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

best_name = comparison.loc[0, "Model"]
best_model = trained_models[best_name]

# Predict using the SAME held-out test set used for comparison
y_pred = best_model.predict(X_test)

# Calculate all required metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    labels=sentiment_labels,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    labels=sentiment_labels,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=sentiment_labels,
    average="macro",
    zero_division=0
)

print("\nFINAL WEEK 2 MODEL EVALUATION")
print("Selected model:", best_name)
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")

report = classification_report(
    y_test,
    y_pred,
    labels=sentiment_labels,
    zero_division=0
)

print("\nClassification Report:")
print(report)

# Save the final confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=sentiment_labels
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=sentiment_labels,
    yticklabels=sentiment_labels
)
plt.title(f"{best_name} - Final Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.savefig(
    "outputs/confusion_matrix_v1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Save a versioned model artifact.
# The TF-IDF vectorizer is included in the saved pipeline.
joblib.dump(
    {
        "pipeline": best_model,
        "model_name": best_name,
        "version": "v1",
        "sentiment_labels": sentiment_labels
    },
    "outputs/sentiment_model_v1.joblib"
)

# Save the evaluation metrics
metrics = {
    "model_name": best_name,
    "version": "v1",
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(macro_f1),
    "selection_metric": "macro_f1",
    "split_method": "GroupShuffleSplit",
    "random_state": 42
}

with open("outputs/evaluation_metrics_v1.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Save detailed report
with open("outputs/classification_report_v1.txt", "w") as file:
    file.write(report)

# Save predictions from the final selected model
predictions_df = pd.DataFrame({
    "review_id": df.loc[X_test.index, "review_id"].values,
    "review_text": X_test.values,
    "actual_sentiment": y_test.values,
    "predicted_sentiment": y_pred
})

predictions_df.to_csv(
    "outputs/test_predictions_v1.csv",
    index=False
)

# Save the model comparison
comparison.to_csv(
    "outputs/model_comparison_v1.csv",
    index=False
)

print("\nFinal model evaluation and artifacts saved successfully.")
print("Selected model:", best_name)
print("Model artifact: outputs/sentiment_model_v1.joblib")
print("Metrics: outputs/evaluation_metrics_v1.json")
print("Confusion matrix: outputs/confusion_matrix_v1.png")
print("Classification report: outputs/classification_report_v1.txt")
print("Test predictions: outputs/test_predictions_v1.csv")
print("Model comparison: outputs/model_comparison_v1.csv")
