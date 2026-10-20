
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("dataset/womens_clothing_analysis.csv")

data = data.dropna(subset=["Review", "Rating"])

data["Sentiment"] = (data["Rating"] >= 4).astype(int)

tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
X = tfidf.fit_transform(data["Review"].astype(str)).toarray()
y = data["Sentiment"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    gamma=0.1,
    eval_metric="logloss",
    early_stopping_rounds=10,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


predictions = model.predict(X_test)

print("Best boosting round:", model.best_iteration)
print("Test Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

results = model.evals_result()

plt.plot(results["validation_0"]["logloss"], label="Validation Loss")
plt.xlabel("Boosting Round")
plt.ylabel("Log Loss")
plt.title("XGBoost Early Stopping")
plt.legend()
plt.show()

for lr in [0.01, 0.05, 0.1]:
    for depth in [3, 5]:
        test_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=lr,
            max_depth=depth,
            subsample=0.8,
            gamma=0.1,
            eval_metric="logloss",
            early_stopping_rounds=10,
            random_state=42
        )

        test_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        score = accuracy_score(y_test, test_model.predict(X_test))

        print(
            "Learning Rate:", lr,
            "Depth:", depth,
            "Accuracy:", round(score, 4),
            "Best Round:", test_model.best_iteration
        )
