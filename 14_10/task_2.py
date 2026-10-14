
import streamlit as st
import pandas as pd

st.header("Womens Clothing Analysis")

file = st.file_uploader("Upload your CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Original Dataset")
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df)

   
    st.subheader("Search Your Data")
    search = st.text_input("Enter your search keyword")

    if search:
        search_result = df[
            df.astype(str).apply(
                lambda row: row.str.contains(
                    search, case=False, na=False
                ).any(),
                axis=1
            )
        ]

        st.write("Search Results:", len(search_result))
        st.dataframe(search_result, use_container_width=True)

    #
    st.subheader("Filter Your Data")
    column = st.selectbox("Select Column to Filter", df.columns)

    value = st.multiselect(
        "Select Values",
        df[column].dropna().unique()
    )

    if value:
        filter_result = df[df[column].isin(value)]
    else:
        filter_result = df.copy()

    st.write("Filtered Records:", len(filter_result))
    st.dataframe(filter_result, use_container_width=True)

    
    st.subheader("Sort Your Data")
    sort_column = st.selectbox("Select Column to Sort", df.columns)

    order = st.selectbox(
        "Select Sort Order",
        ["Ascending", "Descending"]
    )

    sort_result = df.sort_values(
        by=sort_column,
        ascending=(order == "Ascending")
    )

    st.write("Sorted Dataset:", len(sort_result))
    st.dataframe(sort_result, use_container_width=True)

else:
    st.info("Please upload a CSV file.")
