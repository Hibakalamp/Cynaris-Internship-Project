# RetailSense Product Review Sentiment Analysis

## Project Overview

This project analyses product reviews for RetailSense, an e-commerce company. It uses Machine Learning and Natural Language Processing techniques to classify customer reviews into three sentiment categories: **Positive, Negative, and Neutral**.

The project includes a sentiment prediction API and an interactive Streamlit dashboard to explore customer feedback and sentiment trends across product categories.

## Objectives

- Analyse the provided dataset of 1,000 product reviews.
- Classify reviews as positive, negative, or neutral.
- Display overall sentiment distribution.
- Compare sentiment across product categories.
- Provide an interactive dashboard for analysing individual reviews.
- Evaluate the trained model using standard classification metrics.

## Dataset

The dataset contains 1,000 product reviews and 12 columns, including:

- `review_id`: Unique review identifier.
- `product_id`: Product identifier.
- `category`: Product category.
- `rating`: Product rating.
- `review_date`: Date of review.
- `sentiment`: Sentiment label.
- `review_text`: Customer review text.
- `original_sentiment`: Original sentiment information.

The dataset covers categories such as Beauty, Books, Furniture, Home & Kitchen, Clothing, Toys, Grocery, Electronics, Auto, and Sports.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF vectorisation
- Logistic Regression
- FastAPI
- Streamlit
- Plotly
- Joblib

## Project Workflow

1. Load and inspect the product review dataset.
2. Prepare the review text for sentiment analysis.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression classification model.
5. Evaluate the model on held-out test data.
6. Serve sentiment predictions through a FastAPI endpoint.
7. Use Streamlit to display sentiment distribution and category-level analysis.

## Model Evaluation

Previously recorded held-out evaluation results:

Metric Score
Accuracy: 96.62%
Macro Precision: 97.00%
Macro Recall: 79.17%
Macro F1-score: 83.31%

The model exceeds the project's 82% accuracy target based on the recorded evaluation. Confirm that the reported evaluation used the required held-out 200 reviews before final submission.

**Limitation:** Neutral-class recall was 37.5%, indicating that the model needs improvement in identifying neutral reviews.

## Running the Project

### 1. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn fastapi uvicorn streamlit plotly requests joblib
```

### 2. Start the API

From the project directory, run:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

### 3. Start the dashboard

Open a second terminal in the project directory and run:

```bash
python -m streamlit run dashboard.py
```

Open the local URL displayed by Streamlit in your browser.

## API

The dashboard sends review text to the FastAPI prediction endpoint.

- Prediction endpoint: `POST /predict`
- Health endpoint: `GET /health`, if enabled in the API code.
- API documentation: `/docs`

Example request:

```json
{
  "review": "The product quality is excellent."
}
```

## Testing

The project includes an API test script and a CSV file containing API test results. The test script checks API responses for sample reviews.

Successful API requests confirm that the endpoint responds correctly; they do not by themselves establish model classification accuracy.

## Project Structure

```text
Cynaris-Internship-Project/
├── api.py
├── dashboard.py
├── aiml_product_reviews.csv
├── outputs/
│   ├── test_api.py
│   └── api_test_results.csv
└── README.md
```

The trained model artifact should also be included in the project structure if required by the API.

## Future Improvements

- Improve neutral sentiment classification.
- Expand model evaluation and error analysis.
- Improve dashboard insights and visualisations.
- Add more robust handling of missing or invalid reviews.

## Conclusion

This project demonstrates an end-to-end product review sentiment analysis workflow, combining Machine Learning, a REST API, and an interactive dashboard to help explore customer feedback.
