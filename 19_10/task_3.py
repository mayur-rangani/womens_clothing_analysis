
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, pipeline
)

st.title("NLP Sentiment Analysis")

df = pd.read_csv("dataset/womens_clothing_analysis.csv")
df = df.dropna(subset=["Review", "Rating"])
df["label"] = (pd.to_numeric(df["Rating"]) >= 4).astype(int)
df = df.sample(min(500, len(df)), random_state=42)

df["Sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
st.plotly_chart(px.histogram(df, x="Sentiment", color="Sentiment"))

if st.button("Train Model"):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2
    )

    data = Dataset.from_pandas(df[["Review", "label"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["Review"], truncation=True, padding="max_length", max_length=64)

    data = data.map(tokenize, batched=True)
    data = data.remove_columns(["Review"])
    data.set_format("torch")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="results", num_train_epochs=1,
            per_device_train_batch_size=8, report_to="none",
            save_strategy="no"
        ),
        train_dataset=data
    )
    trainer.train()
    model.save_pretrained("sentiment_model")
    tokenizer.save_pretrained("sentiment_model")
    st.success("Model trained and saved!")

if os.path.exists("sentiment_model"):
    @st.cache_resource
    def load_model():
        return pipeline("text-classification", model="sentiment_model")

    text = st.text_area("Enter a review")
    if st.button("Predict"):
        if text.strip():
            result = load_model()(text[:2000])[0]
            sentiment = "Positive" if result["label"] == "LABEL_1" else "Negative"
            st.write("Sentiment:", sentiment)
            st.write("Confidence:", round(result["score"] * 100, 2), "%")
else:
    st.info("Click Train Model first to enable predictions.")
