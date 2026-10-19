import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("CSV Upload and Prediction")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)

        if df.empty or len(df.columns) < 2:
            st.error("CSV must contain data and at least two columns.")
        else:
            st.subheader("Missing Values")
            st.write(df.isnull().sum())

            df = df.dropna()
            df = df.loc[:, df.nunique() > 1]

            st.subheader("Dataset")
            st.dataframe(df)

            st.subheader("Summary Statistics")
            st.write(df.describe())

            columns = df.select_dtypes(include="number").columns

            if len(columns) >= 2 and len(df) >= 5:
                column = st.selectbox("Select Column for Chart", columns)
                fig = px.histogram(df, x=column)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Prediction")

                target = st.selectbox("Select Target Column", columns)
                features = [c for c in columns if c != target]

                if features:
                    X = df[features]
                    y = df[target]

                    model = RandomForestRegressor(
                        n_estimators=100, random_state=42
                    )
                    model.fit(X, y)

                    st.write("Enter values for prediction:")
                    values = []

                    for c in features:
                        values.append(st.number_input(
                            c, value=float(df[c].mean())
                        ))

                    if st.button("Predict"):
                        prediction = model.predict(
                            pd.DataFrame([values], columns=features)
                        )
                        st.success(f"Predicted value: {prediction[0]:.2f}")
        

    except Exception as e:
        st.error(f"Invalid or unsupported CSV data: {e}")