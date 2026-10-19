import pandas as pd
import streamlit as st
import plotly.express as px

st.title("Employee dashboard")

df = pd.read_csv("dataset/womens_clothing_analysis.csv")

st.dataframe(df)

Category=st.multiselect("select your category",df["Category"].dropna().unique(),
                          default=list(df["Category"].dropna().unique()))

if Category:
    department_data = df[df["Category"].isin(Category)]
    st.write("Selected category:", Category)
    st.dataframe(Category)
else:
    department_data = df.copy()
    st.info("No category selected. Showing all category.")


min_price=int(df["Unit_Price_INR"].min())
max_price=int(df["Unit_Price_INR"].max())


price_range=st.slider("select your price range",min_price,max_price,(min_price,max_price))

price_data = df[
    df["Unit_Price_INR"].between(price_range[0],price_range[1])
]

st.write("Selected price Range:", price_range)
st.dataframe(price_data)

search=st.text_input("Product_Name")

search_data=df[df["Product_Name"].str.contains(search,case=False,na=False)]

st.write("product_name:",len(search_data))
st.dataframe(search_data)

st.subheader("4. Dynamic Chart")

filtered_df = df[
    (df["Category"].isin(Category) if Category else True)
    & df["Unit_Price_INR"].between(price_range[0], price_range[1])
    & df["Product_Name"].str.contains(search, case=False, na=False)
]

st.write("Total PRODUCT:", len(filtered_df))

fig = px.bar(filtered_df, x="Category", y="Unit_Price_INR")

st.plotly_chart(fig, use_container_width=True)