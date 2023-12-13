import streamlit as st
from PIL import Image

from utils import load_model, predict_and_nms, visualize_predictions


st.set_page_config(
    page_title="Gradient-free deep learning: Demo",
    page_icon="🕹️",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Demo")

model, device = load_model('data/MOBILE_RCNN.pth')

uploaded_photo = st.file_uploader("Choose a photo")

if uploaded_photo is not None:
    if uploaded_photo.type not in ["image/png", "image/jpeg"]:
        st.error("This is not an image file (must be jpeg or png).")
    else:
        image = Image.open(uploaded_photo).convert('RGB')
        st.image(
            image, caption="Uploaded Image.", use_column_width=True
        )
        if st.button('Predict'):
            boxes, labels, scores = predict_and_nms(model, image, device)
            result_image = visualize_predictions(image, boxes, labels,
            {
                0: 'vehicles',
                1: 'big bus',
                2: 'big truck',
                3: 'bus-l-',
                4: 'bus-s-',
                5: 'car',
                6: 'mid truck',
                7: 'small bus',
                8: 'small truck',
                9: 'truck-l-',
                10: 'truck-m-',
                11: 'truck-s-',
                12: 'truck-xl-',
            })

            # Display the result image
            st.image(result_image, caption='Predicted Image', use_column_width=True)

            # Display results
            for i in range(len(boxes)):
                st.write(f"Label: {labels[i]}, Score: {scores[i]:.2f}")
        