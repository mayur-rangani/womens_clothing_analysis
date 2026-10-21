
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


data = pd.read_csv("dataset/womens_clothing_analysis.csv")
data = data.dropna(subset=["Review", "Rating"])
data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce")
data = data.dropna(subset=["Rating"])

X = data["Review"].astype(str)
y = (data["Rating"] >= 4).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
    ("lr", LogisticRegression(C=1, max_iter=1000))
])


sizes, train, valid = learning_curve(
    model, X_train, y_train, cv=3, scoring="accuracy",
    train_sizes=[0.3, 0.6, 1.0]
)

plt.plot(sizes, train.mean(axis=1), label="Training")
plt.plot(sizes, valid.mean(axis=1), label="Validation")
plt.xlabel("Training Data Size")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curve")
plt.show()

for penalty in ["l1", "l2"]:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
        ("lr", LogisticRegression(
            penalty=penalty, solver="liblinear", max_iter=1000
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(penalty, "Accuracy:", accuracy_score(y_test, pred),
          "F1:", f1_score(y_test, pred))

selector = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
    ("select", SelectFromModel(
        LogisticRegression(penalty="l1", solver="liblinear")
    )),
    ("lr", LogisticRegression(max_iter=1000))
])
selector.fit(X_train, y_train)
pred = selector.predict(X_test)
print("Feature Selection F1:", f1_score(y_test, pred))

ensemble = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
    ("vote", VotingClassifier([
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42))
    ], voting="soft"))
])
ensemble.fit(X_train, y_train)
pred = ensemble.predict(X_test)
print("Ensemble Accuracy:", accuracy_score(y_test, pred))
print("Ensemble F1:", f1_score(y_test, pred))
