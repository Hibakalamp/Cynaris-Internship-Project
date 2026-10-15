
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from pathlib import Path


DATASET_PATH = Path(__file__).parent / "aiml_product_reviews_corrected.csv"

@st.cache_data(show_spinner=False)
def load_review_dataset(csv_path, modified_time):
    df = pd.read_csv(csv_path)

    required_columns = {"category", "review_text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["category", "review_text"]).copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[
        (df["category"] != "") & (df["review_text"] != "")
    ].copy()

    return df


@st.cache_data(show_spinner=False)
def predict_dataset_sentiments(csv_path, modified_time, api_url):
    df = load_review_dataset(csv_path, modified_time)

    predictions = []

    for review in df["review_text"]:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json={"review": review},
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        sentiment = result.get("predicted_sentiment")

        if sentiment not in {"positive", "negative", "neutral"}:
            raise ValueError(
                f"Unexpected sentiment returned by API: {sentiment}"
            )

        predictions.append(sentiment)

    df["predicted_sentiment"] = predictions
    return df
    
# ---------------- CONFIGURATION ----------------
API_URL = "http://127.0.0.1:8001"
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"

PRODUCT_CATEGORIES = [
    "Beauty",
    "Books",
    "Furniture",
    "Home & Kitchen",
    "Clothing",
    "Toys",
    "Grocery",
    "Electronics",
    "Auto",
    "Sports",
]
SENTIMENT_COLORS = {
    "Positive": "#16A36A",
    "Negative": "#E45460",
    "Neutral": "#8B83F6",
}

st.set_page_config(
    page_title="RetailSense | Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F5F7FB;
    }

    [data-testid="stSidebar"] {
        background: #17243A;
    }

    [data-testid="stSidebar"] * {
        color: #F1F5F9;
    }

    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(120deg, #17243A, #334D79);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 7px;
    }

    .hero p {
        color: #DCE6F5;
        margin: 0;
    }

    .metric-card {
        background: white;
        padding: 21px;
        border-radius: 15px;
        border: 1px solid #E7EAF1;
        box-shadow: 0 3px 12px rgba(20, 35, 60, 0.04);
        min-height: 125px;
    }

    .metric-label {
        color: #65738A;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #17243A;
        font-size: 28px;
        font-weight: 800;
        margin-top: 9px;
    }

    .metric-note {
        color: #78869B;
        font-size: 12px;
        margin-top: 5px;
    }

    .section-heading {
        color: #17243A;
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #78869B;
        font-size: 13px;
        margin-bottom: 17px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #465FEA;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #354BD0;
        color: white;
        border: none;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
        border: 1px solid #DCE2EC;
    }

    div[data-testid="stSelectbox"] > div {
        border-radius: 9px;
    }

    .footer {
        text-align: center;
        color: #8A96A8;
        font-size: 12px;
        padding: 25px 0 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SESSION DATA ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HELPERS ----------------
def predict_sentiment(review):
    """Send a review to the existing FastAPI model."""
    response = requests.post(
        PREDICT_URL,
        json={"review": review},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    sentiment = str(data["predicted_sentiment"]).lower().strip()

    if sentiment not in ["positive", "negative", "neutral"]:
        raise ValueError(
            f"Unexpected sentiment label: {sentiment}"
        )

    return sentiment


def save_result(review, sentiment, category):
    """Save one prediction to the current Streamlit session."""
    st.session_state.history.append(
        {
            "Time": datetime.now().strftime("%d %b %Y, %H:%M"),
            "Category": category,
            "Review": review,
            "Sentiment": sentiment.title(),
        }
    )


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sentiment_result(sentiment):
    if sentiment == "positive":
        st.success(
            "😊 POSITIVE — The review expresses favourable sentiment."
        )
    elif sentiment == "negative":
        st.error(
            "☹️ NEGATIVE — The review expresses unfavourable sentiment."
        )
    else:
        st.warning(
            "😐 NEUTRAL — The review expresses neutral sentiment."
        )


def get_history_df():
    columns = ["Time", "Category", "Review", "Sentiment"]
    return pd.DataFrame(
        st.session_state.history,
        columns=columns,
    )


def check_api():
    try:
        response = requests.get(HEALTH_URL, timeout=2)
        return response.ok
    except requests.RequestException:
        return False


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# RetailSense")
    st.caption("CUSTOMER INTELLIGENCE")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Review Analyzer",
            "Analysis History",
            "About Project",
        ],
    )

    st.markdown("---")
    st.markdown("**MODEL INFORMATION**")
    st.caption("Algorithm: Logistic Regression")
    st.caption("Features: TF-IDF")
    st.caption("Model version: v1")

    api_online = check_api()

    if api_online:
        st.success("API connection active")
    else:
        st.error("API offline")
        st.caption("Start FastAPI on port 8001.")

    st.markdown("---")
    st.caption("RetailSense Pvt Ltd")
    st.caption("AI/ML Internship Project")

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1>Customer Review Intelligence</h1>
        <p>
            Understand customer opinions with AI-powered
            sentiment analysis and product-category insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- OVERVIEW ----------------
st.subheader("Full Dataset Sentiment Analysis")

if not DATASET_PATH.exists():
    st.error(f"Dataset not found: {DATASET_PATH.name}")
else:
    try:
        modified_time = DATASET_PATH.stat().st_mtime

        with st.spinner(
            "Analysing dataset reviews through the sentiment API..."
        ):
            dataset_df = predict_dataset_sentiments(
                str(DATASET_PATH),
                modified_time,
                API_URL,
            )

        st.caption(
            f"Showing model predictions for {len(dataset_df):,} "
            "valid reviews in the provided dataset."
        )

        # Overall sentiment distribution
        sentiment_counts = (
            dataset_df["predicted_sentiment"]
            .value_counts()
            .reindex(
                ["positive", "neutral", "negative"],
                fill_value=0,
            )
            .rename_axis("Sentiment")
            .reset_index(name="Review Count")
        )

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Review Count",
            title="Overall Sentiment Distribution",
            hole=0.55,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment breakdown by category
        category_counts = (
            dataset_df.groupby(
                ["category", "predicted_sentiment"]
            )
            .size()
            .reset_index(name="Review Count")
        )

        fig = px.bar(
            category_counts,
            x="category",
            y="Review Count",
            color="predicted_sentiment",
            barmode="group",
            category_orders={
                "predicted_sentiment": [
                    "positive", "neutral", "negative"
                ]
            },
            title="Sentiment Breakdown by Product Category",
            labels={
                "category": "Product Category",
                "predicted_sentiment": "Predicted Sentiment",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        # Category summary table
        category_summary = pd.crosstab(
            dataset_df["category"],
            dataset_df["predicted_sentiment"],
        ).reindex(
            columns=["positive", "neutral", "negative"],
            fill_value=0,
        )

        category_summary["Total Reviews"] = category_summary.sum(axis=1)

        st.subheader("Category-wise Summary")
        st.dataframe(category_summary, use_container_width=True)

    except requests.RequestException as exc:
        st.error(
            "Could not get sentiment predictions. "
            "Make sure the FastAPI server is running."
        )
        st.caption(str(exc))

    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"Dataset analysis failed: {exc}")



if page == "Overview":
    st.markdown("### Sentiment Overview")
    st.caption(
        "Summary of customer reviews analysed in this session."
    )

    history_df = get_history_df()

    if not history_df.empty:
        counts = history_df["Sentiment"].value_counts()
        positive = int(counts.get("Positive", 0))
        negative = int(counts.get("Negative", 0))
        neutral = int(counts.get("Neutral", 0))
        total = len(history_df)
    else:
        positive = negative = neutral = total = 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "REVIEWS ANALYSED", total, "Current session"
        )
    with c2:
        metric_card(
            "POSITIVE REVIEWS", positive, "Favourable feedback"
        )
    with c3:
        metric_card(
            "NEGATIVE REVIEWS", negative, "Issues to investigate"
        )
    with c4:
        metric_card(
            "NEUTRAL REVIEWS", neutral, "Neutral feedback"
        )

    st.write("")

    # Overall distribution and quick analyzer
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">'
                'Sentiment Distribution</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">'
                'Breakdown of your analysed reviews</div>',
                unsafe_allow_html=True,
            )

            if total > 0:
                chart_df = pd.DataFrame(
                    {
                        "Sentiment": [
                            "Positive",
                            "Negative",
                            "Neutral",
                        ],
                        "Count": [positive, negative, neutral],
                    }
                )
                chart_df = chart_df[chart_df["Count"] > 0]

                fig = px.pie(
                    chart_df,
                    names="Sentiment",
                    values="Count",
                    hole=0.65,
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                )

                fig.update_layout(
                    showlegend=True,
                    legend_title_text="",
                    margin=dict(t=10, b=10, l=5, r=5),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#35445B"),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )
            else:
                st.info(
                    "Analyse a review to see your sentiment chart."
                )

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">'
                'Quick Review Analysis</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">'
                'Choose a category and predict review sentiment'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.form("quick_review_form"):
                category = st.selectbox(
                    "Product category",
                    PRODUCT_CATEGORIES,
                    key="quick_category",
                )

                review = st.text_area(
                    "Customer review",
                    placeholder=(
                        "Example: The quality is excellent "
                        "and delivery was fast..."
                    ),
                    height=140,
                )

                submitted = st.form_submit_button(
                    "✦  Analyse Sentiment",
                    use_container_width=True,
                )

            if submitted:
                if not review.strip():
                    st.warning("Please enter a customer review.")
                elif not api_online:
                    st.error(
                        "The API is offline. Start FastAPI "
                        "on port 8001 and try again."
                    )
                else:
                    try:
                        with st.spinner("Analysing review..."):
                            sentiment = predict_sentiment(
                                review.strip()
                            )

                        save_result(
                            review.strip(),
                            sentiment,
                            category,
                        )
                        show_sentiment_result(sentiment)

                    except requests.RequestException:
                        st.error(
                            "Could not connect to the API. "
                            "Check that FastAPI is running."
                        )
                    except (ValueError, KeyError) as exc:
                        st.error(f"Prediction error: {exc}")

    # -------- PRODUCT CATEGORY SENTIMENT ANALYSIS --------
    st.write("")
    st.markdown("### Product Category Insights")
    st.caption(
        "Compare positive, negative and neutral feedback "
        "across product categories."
    )

    history_df = get_history_df()

    if not history_df.empty:
        category_counts = (
            history_df.groupby(
                ["Category", "Sentiment"]
            )
            .size()
            .reset_index(name="Reviews")
        )

        chart_col, summary_col = st.columns(
            [1.2, 0.8],
            gap="large",
        )

        with chart_col:
            with st.container(border=True):
                st.markdown("#### Sentiment by Category")

                category_fig = px.bar(
                    category_counts,
                    x="Category",
                    y="Reviews",
                    color="Sentiment",
                    barmode="group",
                    color_discrete_map=SENTIMENT_COLORS,
                    category_orders={
                        "Sentiment": [
                            "Positive",
                            "Negative",
                            "Neutral",
                        ]
                    },
                    labels={
                        "Category": "Product Category",
                        "Reviews": "Number of Reviews",
                        "Sentiment": "Sentiment",
                    },
                )

                category_fig.update_layout(
                    height=420,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#35445B"),
                    legend_title_text="Sentiment",
                    margin=dict(
                        t=20, b=100, l=15, r=15
                    ),
                )

                category_fig.update_xaxes(
                    tickangle=-25,
                    showgrid=False,
                )
                category_fig.update_yaxes(
                    rangemode="tozero",
                    gridcolor="#EDF0F5",
                )

                st.plotly_chart(
                    category_fig,
                    use_container_width=True,
                )

        with summary_col:
            with st.container(border=True):
                st.markdown("#### Category Summary")

                category_summary = (
                    history_df.pivot_table(
                        index="Category",
                        columns="Sentiment",
                        values="Review",
                        aggfunc="count",
                        fill_value=0,
                    )
                    .reindex(
                        columns=[
                            "Positive",
                            "Negative",
                            "Neutral",
                        ],
                        fill_value=0,
                    )
                    .reset_index()
                )

                category_summary["Total Reviews"] = (
                    category_summary["Positive"]
                    + category_summary["Negative"]
                    + category_summary["Neutral"]
                )

                st.dataframe(
                    category_summary,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Counts reflect reviews analysed in this session."
                )
    else:
        st.info(
            "Analyse reviews and select their product categories "
            "to view category-level sentiment insights."
        )

    # -------- RECENT ANALYSES --------
    st.write("")
    st.markdown("### Recent Analyses")

    if not history_df.empty:
        st.dataframe(
            history_df.tail(5).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(
            "Your recent analyses will appear here."
        )

# ---------------- REVIEW ANALYZER ----------------
elif page == "Review Analyzer":
    st.markdown("### Review Analyzer")
    st.caption(
        "Submit a customer review and select its product category."
    )

    with st.container(border=True):
        category = st.selectbox(
            "Product category",
            PRODUCT_CATEGORIES,
            key="analyzer_category",
        )

        review = st.text_area(
            "Product review",
            placeholder=(
                "Write or paste the full customer review here..."
            ),
            height=180,
        )

        if st.button(
            "Analyse Review",
            use_container_width=True,
        ):
            if not review.strip():
                st.warning("Please enter a review first.")
            elif not api_online:
                st.error(
                    "The API is offline. Start FastAPI on port 8001."
                )
            else:
                try:
                    with st.spinner(
                        "Sending review to the model..."
                    ):
                        sentiment = predict_sentiment(
                            review.strip()
                        )

                    save_result(
                        review.strip(),
                        sentiment,
                        category,
                    )

                    st.markdown("#### Prediction Result")
                    show_sentiment_result(sentiment)

                    st.markdown("**Product category**")
                    st.write(category)

                    st.markdown("**Submitted review**")
                    st.write(review.strip())

                except requests.RequestException:
                    st.error(
                        "API connection failed. "
                        "Start the FastAPI server and try again."
                    )
                except (ValueError, KeyError) as exc:
                    st.error(f"Prediction error: {exc}")

    st.caption(
        "Predictions come from your trained model and may not "
        "always be correct. The category is selected separately."
    )

# ---------------- ANALYSIS HISTORY ----------------
elif page == "Analysis History":
    st.markdown("### Analysis History")
    st.caption(
        "Reviews analysed during the current dashboard session."
    )

    history_df = get_history_df()

    if not history_df.empty:
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download History as CSV",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="retailsense_analysis_history.csv",
            mime="text/csv",
        )

        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No reviews analysed yet. Visit Review Analyzer to get started."
        )

# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":
    st.markdown("### About RetailSense")

    st.write(
        "RetailSense is an AI/ML-based sentiment analysis project "
        "designed to evaluate customer feedback in the e-commerce "
        "industry. Using Natural Language Processing (NLP) and "
        "machine learning, the system classifies product reviews "
        "as positive, negative, or neutral. Product categories are "
        "recorded separately to help compare customer sentiment "
        "across different product groups."
    )

    st.markdown("#### Model Pipeline")

    st.markdown(
        """
        1. **Input:** Customer product review.
        2. **Category:** Select the relevant product category.
        3. **Text processing:** TF-IDF feature extraction.
        4. **Classification:** Logistic Regression model.
        5. **Output:** Predicted positive, negative, or neutral label.
        6. **Dashboard:** Display overall and category-level sentiment.
        """
    )

    st.markdown("#### Held-out Model Evaluation")

    a, b, c = st.columns(3)

    with a:
        metric_card(
            "ACCURACY",
            "96.62%",
            "Held-out test set",
        )
    with b:
        metric_card(
            "MACRO F1",
            "83.31%",
            "Average across classes",
        )
    with c:
        metric_card(
            "NEUTRAL RECALL",
            "37.50%",
            "Area for improvement",
        )

    st.warning(
        "These metrics are from the recorded held-out evaluation, "
        "not live dashboard performance. Neutral recall is low, "
        "so some neutral reviews may be misclassified."
    )

    st.markdown("#### Current Limitations")

    st.markdown(
        """
        - Category is selected by the user; the sentiment model
          does not predict the category.
        - Category charts reflect reviews analysed in the current
          session.
        - Session history is not a permanent database.
        - Model predictions can be incorrect, particularly for
          neutral reviews.
        """
    )

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        RetailSense Sentiment Analytics · AI/ML Internship Project 
    </div>
    """,
    unsafe_allow_html=True,
)
