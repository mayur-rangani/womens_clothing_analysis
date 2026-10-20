
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

data = pd.read_csv("dataset/womens_clothing_analysis.csv")
data = data.dropna(subset=["Review", "Rating"])

X = data["Review"].astype(str)
y = (data["Rating"] >= 4).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english")),
    ("xgb", XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])

model.fit(X_train, y_train)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/sentiment_model.joblib", compress=3)

print("Model saved successfully!")
print("Test accuracy:", model.score(X_test, y_test))
