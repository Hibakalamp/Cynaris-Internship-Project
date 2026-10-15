
import pandas as pd

df = pd.read_csv("aiml_product_reviews_corrected.csv")

# Clean review text and sentiment labels
df["review_text"] = (
    df["review_text"].fillna("").astype(str).str.strip()
)
df["sentiment"] = (
    df["sentiment"].fillna("").astype(str).str.strip().str.lower()
)

# Normalize text for duplicate checking
df["normalized_text"] = (
    df["review_text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

print("Total rows:", len(df))
print("Unique review texts:", df["normalized_text"].nunique())

print("\nRows per sentiment:")
print(df["sentiment"].value_counts())

# Check whether the same review text has different labels
label_counts = df.groupby("normalized_text")["sentiment"].nunique()
conflicting_texts = label_counts[label_counts > 1]

print("\nReview texts with conflicting sentiment labels:",
      len(conflicting_texts))

# Count distinct review texts for each sentiment
unique_pairs = df.drop_duplicates(
    subset=["normalized_text", "sentiment"]
)

print("\nUnique text-label pairs per sentiment:")
print(unique_pairs["sentiment"].value_counts())

print("\nSample unique reviews:")
print(
    unique_pairs[["review_text", "sentiment"]]
    .head(15)
    .to_string(index=False)
)

import pandas as pd

df = pd.read_csv("aiml_product_reviews.csv")

print("Original dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique review texts:", df["review_text"].nunique())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts(dropna=False))

print("\nSample reviews:")
print(
    df[["review_text", "sentiment"]]
    .drop_duplicates(subset=["review_text"])
    .head(20)
    .to_string(index=False)
)
