import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="my page",
    layout="wide",
    initial_sidebar_state="auto")

st.sidebar.title("my_home_page")

menu=st.sidebar.selectbox("select menu",["home","dashboard","chart","about"])

if menu=="home":
    st.title("welcome to home page")
    st.write("this is home page")

elif menu=="dashboard":
    st.title("welcome to dashboard")
    st.write("this is dashboard")

elif menu=="chart":
    st.title("welcome visulization page")        
    st.write("this is chart page")

elif menu=="about":
    st.title("this is about page")
    st.write("information about project")    