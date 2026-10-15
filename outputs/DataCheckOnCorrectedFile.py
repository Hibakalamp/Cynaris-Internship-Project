
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("aiml_product_reviews_corrected.csv")

print("Total rows:", len(df))
print("Unique review texts:", df["review_text"].nunique())

# Check whether each text has exactly one sentiment label
labels_per_text = df.groupby("review_text")["sentiment"].nunique()

print(
    "Texts with multiple sentiment labels:",
    (labels_per_text > 1).sum()
)

# Recreate a stratified split
X_train, X_test, y_train, y_test = train_test_split(
    df["review_text"],
    df["sentiment"],
    test_size=0.20,
    random_state=42,
    stratify=df["sentiment"]
)

# Check overlap between train and test review texts
train_texts = set(X_train)
test_texts = set(X_test)

overlap = train_texts.intersection(test_texts)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Unique texts appearing in both sets:", len(overlap))
print("Total unique test texts:", len(test_texts))

