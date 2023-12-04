import streamlit as st

from PIL import Image


st.set_page_config(
    page_title="Gradient-free deep learning: Demo",
    page_icon="🕹️",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Demo")

uploaded_photo = st.file_uploader("Choose a photo")

if uploaded_photo is not None:
    if uploaded_photo.type not in ["image/png", "image/jpeg"]:
        st.error("This is not an image file (must be jpeg or png).")
    else:
        st.image(
            Image.open(uploaded_photo), caption="Uploaded Image.", use_column_width=True
        )
