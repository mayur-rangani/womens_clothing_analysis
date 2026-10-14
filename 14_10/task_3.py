
import streamlit as st
import pandas as pd

st.title("Womens Clothing Analysis")


file = st.file_uploader("Upload your CSV file", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Original Dataset")
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df, use_container_width=True)

    st.subheader("Filter Your Data")

    column = st.selectbox("Select Column to Filter", df.columns)

    values = st.multiselect(
        "Select Values",
        df[column].dropna().unique()
    )

    result = df.copy()

    if values:
        result = result[result[column].isin(values)]

    st.write("Filtered Records:", len(result))
    st.dataframe(result, use_container_width=True)

    
    st.subheader("Sort Your Data")

    sort_column = st.selectbox("Select Column to Sort", df.columns)

    order = st.selectbox(
        "Sort Order",
        ["Ascending", "Descending"]
    )

    sort_result = result.sort_values(
        by=sort_column,
        ascending=(order == "Ascending")
    )

    st.write("Sorted Records:", len(sort_result))
    st.dataframe(sort_result, use_container_width=True)

    
    st.subheader("Visualization")

    chart_column = st.selectbox(
        "Select Column for Visualization",
        sort_result.columns
    )

    if not sort_result.empty:
        if sort_result[chart_column].dtype == "object":
            st.bar_chart(sort_result[chart_column].value_counts())
        else:
            st.bar_chart(
                sort_result[chart_column].value_counts().sort_index()
            )
    else:
        st.warning("No data available for visualization.")

    
    st.subheader("User Input Prediction")

    user_rating = st.slider(
        "Enter Rating for Prediction",
        min_value=1,
        max_value=5,
        value=3
    )

    if user_rating >= 4:
        st.success("Prediction: Positive Review")
    elif user_rating == 3:
        st.info("Prediction: Neutral Review")
    else:
        st.warning("Prediction: Negative Review")

else:
    st.info("Please upload a CSV file.")
