
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


data = pd.read_csv("dataset/womens_clothing_analysis.csv")
data = data.dropna(subset=["Review", "Rating"])
data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce")
data = data.dropna(subset=["Rating"])


review = data["Review"].astype(str)
X = pd.DataFrame({
    "Length": review.str.len(),
    "Words": review.str.split().str.len(),
    "Exclamation": review.str.count("!"),
    "PositiveWords": review.str.lower().str.count(
        r"\b(good|great|excellent|love|comfortable)\b"
    ),
    "NegativeWords": review.str.lower().str.count(
        r"\b(bad|poor|terrible|hate|disappointed)\b"
    )
})

y = (data["Rating"] >= 4).astype(int)


poly = PolynomialFeatures(degree=2, include_bias=False)
X = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("PCA features:", X_train.shape[1])


search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    {"C": [0.1, 1, 10]},
    cv=3, scoring="f1", n_jobs=-1
)
search.fit(X_train, y_train)

model = search.best_estimator_
pred = model.predict(X_test)

print("Best parameters:", search.best_params_)
print("Accuracy:", accuracy_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))

explainer = shap.LinearExplainer(model, X_train[:100])
shap_values = explainer(X_test[:100])

shap.plots.bar(shap_values, max_display=10)

