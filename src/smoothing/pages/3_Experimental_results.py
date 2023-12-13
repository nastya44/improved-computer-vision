import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Gradient-free deep learning: Experimental results",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Experimental results")
st.header("Train loss")
st.line_chart(pd.read_csv("./data/losses.csv"))

st.header("Train accuracy")
st.line_chart(pd.read_csv("./data/accuracy.csv"))

