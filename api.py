from fastapi import FastAPI
app = FastAPI(
    title="RetailSense Sentiment Analysis API",
    version="1.0.0"
)


from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="RetailSense Sentiment Analysis API",
    version="1.0.0",
    description="Predict sentiment from product reviews."
)

# Find the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    BASE_DIR / "sentiment_model_v1.joblib",
    BASE_DIR / "outputs" / "sentiment_model_v1.joblib",
]

model_path = next((p for p in MODEL_PATHS if p.exists()), None)
model = None

if model_path:
    artifact = joblib.load(model_path)
model = artifact["pipeline"]


class ReviewRequest(BaseModel):
    review: str = Field(min_length=1, max_length=5000)


@app.get("/")
def home():
    return {"message": "RetailSense Sentiment Analysis API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_found",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Check the model file path."
        )

    review = request.review.strip()
    if not review:
        raise HTTPException(status_code=422, detail="Review cannot be empty.")

    try:
       
        prediction = model.predict([review])[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        if isinstance(prediction, (int, float)):
            labels = artifact.get("sentiment_labels", [])
            prediction = labels[int(prediction)]

        return {
            "review": review,
            "predicted_sentiment": str(prediction),
        }


    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc
