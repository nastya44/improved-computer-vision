import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Gradient-free deep learning: Experimental results",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Experimental results")
st.header("Test loss")
df = pd.read_excel("./data/results.xlsx")
df.set_index('Method', inplace=True)
st.write(df)

st.header("Test accuracy")
df = pd.read_excel("./data/accuracy.xlsx")
df.set_index('Method', inplace=True)
st.write(df)
