
import pandas as pd

df = pd.read_csv("aiml_product_reviews.csv")

# Check how many unique review texts exist
print("Unique review texts:", df["review_text"].nunique())

# Check whether the same review text has multiple sentiment labels
label_check = (
    df.groupby("review_text")["sentiment"]
    .agg(["count", "nunique", "unique"])
    .sort_values("nunique", ascending=False)
)

print("\nReview text and sentiment consistency:")
print(label_check.to_string())

print("\nSentiment distribution for each review text:")
print(pd.crosstab(df["review_text"], df["sentiment"]))

# Check whether review text and sentiment labels are consistent
inconsistent = label_check[label_check["nunique"] > 1]
print("\nReview texts with conflicting labels:")
print(inconsistent.to_string())

# Check if the three model predictions are really identical
# Use this only after saving each model's predictions separately.
# print((dt_predictions == rf_predictions).all())
# print((dt_predictions == lr_predictions).all())


# Normalize text for matching
df["review_text"] = df["review_text"].fillna("").str.strip()

# Manually reviewed labels based on the review meaning
sentiment_mapping = {
    "Average product. Packaging could be better.": "neutral",
    "Best purchase this year. Highly recommended!": "positive",
    "Damaged on arrival. Replacement was sent promptly.": "negative",
    "Disappointed with quality. Not worth the price.": "negative",
    "Excellent! Exceeded expectations. Will buy again.": "positive",
    "Good quality but slightly expensive. Would recommend.": "positive",
    "Great product, exactly as described. Fast delivery!": "positive",
    "Works perfectly. Customer support was helpful.": "positive"
}

# Check that every text has a reviewed label
unmapped = set(df["review_text"].unique()) - set(sentiment_mapping)

if unmapped:
    print("Review texts needing manual review:", unmapped)
else:
    # Keep the original labels for audit purposes
    df["original_sentiment"] = df["sentiment"]
    df["sentiment"] = df["review_text"].map(sentiment_mapping)

    print("Corrected sentiment distribution:")
    print(df["sentiment"].value_counts())

    df.to_csv("aiml_product_reviews_corrected.csv", index=False)
    print("Corrected dataset saved.")

    
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

