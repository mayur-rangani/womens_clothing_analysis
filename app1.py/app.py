
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Women's Clothing Sentiment",
      layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_model.joblib")

st.title("Women's Clothing Sentiment Analysis")
st.write("Predict whether a clothing review is positive or negative.")

model_path = Path("models/sentiment_model.joblib")

if not model_path.exists():
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

model = load_model()

review = st.text_area("Enter a clothing review")

if st.button("Predict Sentiment"):
    if review.strip():
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]

        if prediction == 1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")

        st.metric(
            "Prediction Confidence",
            f"{max(probabilities) * 100:.2f}%"
        )
    else:
        st.warning("Please enter a review.")

st.caption("Model predictions are estimates, not guarantees.")
