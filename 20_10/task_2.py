
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


data = pd.read_csv("dataset/womens_clothing_analysis.csv")
data = data.dropna(subset=["Review", "Rating"])

data["Sentiment"] = (data["Rating"] >= 4).astype(int)

X = data["Review"].astype(str)
y = data["Sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english")),
    ("model", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0]
}

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)


grid = GridSearchCV(
    pipeline,
    params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)


print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

predictions = grid.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


results = pd.DataFrame(grid.cv_results_)
results["Parameters"] = results["params"].apply(
    lambda p: f"Depth={p['model__max_depth']}, "
              f"LR={p['model__learning_rate']}, "
              f"Trees={p['model__n_estimators']}, "
              f"Sample={p['model__subsample']}"
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=results,
    x="Parameters",
    y="mean_test_score"
)
plt.xticks(rotation=90)
plt.ylabel("Mean Cross-Validation Accuracy")
plt.title("GridSearchCV Parameter Comparison")
plt.tight_layout()
plt.show()
