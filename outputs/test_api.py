
import csv
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:8001/predict"

reviews = [
    "The packaging was damaged and the product arrived broken.",
    "Excellent quality and very good value for money.",
    "The item is okay, nothing special about it.",
    "I am disappointed with this purchase and would not recommend it.",
    "The product works perfectly and exceeded my expectations.",
    "Average product, but delivery was on time.",
    "Poor quality material and the stitching came apart.",
    "I love this product and would definitely buy it again.",
    "The product is acceptable for the price paid.",
    "Completely satisfied with the quality and performance.",
]

results = []

for i, review in enumerate(reviews, start=1):
    try:
        response = requests.post(
            API_URL,
            json={"review": review},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        result = {
            "test_id": i,
            "review": review,
            "predicted_sentiment": data["predicted_sentiment"],
            "status": "PASS",
        }
    except Exception as exc:
        result = {
            "test_id": i,
            "review": review,
            "predicted_sentiment": "",
            "status": f"FAIL: {exc}",
        }

    results.append(result)
    print(f"Test {i}: {result['status']} | {result['predicted_sentiment']}")

output_dir = Path(__file__).resolve().parent
output_file = output_dir / "api_test_results.csv"

with output_file.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["test_id", "review", "predicted_sentiment", "status"],
    )
    writer.writeheader()
    writer.writerows(results)

successful = sum(r["status"] == "PASS" for r in results)
print(f"\nResults saved to: {output_file}")
print(f"Successful API requests: {successful}/10")

