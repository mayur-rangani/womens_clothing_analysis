
import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from bokeh.plotting import figure
from bokeh.models import HoverTool
from streamlit_bokeh import streamlit_bokeh

st.title("Data Visualization")

file = st.file_uploader("Upload your CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset")
    st.dataframe(df)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) >= 2:

        x_col = st.selectbox(
            "Select X-axis",
            numeric_columns
        )

        y_col = st.selectbox(
            "Select Y-axis",
            numeric_columns,
            index=1
        )

        library = st.selectbox(
            "Choose your library",
            ["plotly", "altair", "bokeh"]
        )

       
        if library == "plotly":

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                hover_data=df.columns.to_list(),
                title="Plotly Scatter Plot"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

      
        elif library == "altair":

            chart = alt.Chart(df).mark_bar(
                size=100
            ).encode(
                x=alt.X(
                    x_col,
                    title=x_col
                ),
                y=alt.Y(
                    y_col,
                    title=y_col
                ),
                tooltip=df.columns.to_list()
            ).interactive()

            st.altair_chart(
                chart,
                use_container_width=True
            )

        
        elif library == "bokeh":

            p = figure(
                title="Bokeh Scatter Plot",
                x_axis_label=x_col,
                y_axis_label=y_col,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                height=450,
                sizing_mode="stretch_width"
            )

            p.scatter(
                x=df[x_col],
                y=df[y_col],
                size=10,
                legend_label="Data Points"
            )

            hover = HoverTool(
                tooltips=[
                    (x_col, "@x"),
                    (y_col, "@y")
                ]
            )

            p.add_tools(hover)

            streamlit_bokeh(p)


    else:
        st.warning(
            "CSV must contain at least two numeric columns."
        )

else:
    st.info("Please upload a CSV file to visualize your data.")
