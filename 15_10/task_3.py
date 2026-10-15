
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Model Prediction App", layout="wide")

st.title("Model Prediction and Explanation")
st.write("Select a model, enter feature values, and view its prediction.")

data = pd.read_csv("dataset/womens_clothing_analysis.csv")

st.subheader("Dataset Preview")
st.write("Dataset Shape:", data.shape)
st.dataframe(data.head())

target_column = st.selectbox(
    "Select Target Column",
    data.columns
)

data = data.dropna(subset=[target_column]).copy()

X = data.drop(columns=[target_column])
y = data[target_column]

X = X.select_dtypes(include="number")

valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

if X.empty or X.shape[1] == 0:
    st.error("No usable numeric feature columns found.")
    st.stop()

if y.nunique() < 2:
    st.error("The target column must contain at least two classes.")
    st.stop()

class_labels = sorted(y.unique(), key=str)
label_to_number = {
    label: i for i, label in enumerate(class_labels)
}
y = y.map(label_to_number)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )
}

for model in models.values():
    model.fit(X_train, y_train)


model_name = st.selectbox(
    "Choose a Model",
    list(models.keys())
)

model = models[model_name]

st.subheader("Enter Feature Values")


user_values = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(X.columns):
    with (col1 if i % 2 == 0 else col2):
        user_values[feature] = st.number_input(
            feature,
            value=float(X[feature].median()),
            format="%.4f"
        )

input_data = pd.DataFrame([user_values], columns=X.columns)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    predicted_class = class_labels[int(prediction)]

    st.subheader("Prediction Result")
    st.success(f"Predicted class: {predicted_class}")

    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities[list(model.classes_).index(prediction)] * 100

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    probability_df = pd.DataFrame({
        "Class": [
            str(class_labels[int(c)]) for c in model.classes_
        ],
        "Probability (%)": probabilities * 100
    })

    st.subheader("Class Probabilities")
    st.bar_chart(probability_df.set_index("Class"))

    if model_name == "Random Forest":
        st.subheader("SHAP Feature Explanation")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)

            if isinstance(shap_values, list):
                class_index = list(model.classes_).index(prediction)
                values = shap_values[class_index][0]
            elif len(shap_values.shape) == 3:
                class_index = list(model.classes_).index(prediction)
                values = shap_values[0, :, class_index]
            else:
                values = shap_values[0]

            explanation_df = pd.DataFrame({
                "Feature": X.columns,
                "SHAP Value": values
            })

            explanation_df["Absolute Value"] = (
                explanation_df["SHAP Value"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute Value", ascending=False
            ).head(10)

            st.write("Top features influencing the prediction:")
            st.bar_chart(
                explanation_df.set_index("Feature")["SHAP Value"]
            )

        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

    else:
        st.info(
            "SHAP explanation is currently implemented "
            "for the Random Forest model only."
        )

