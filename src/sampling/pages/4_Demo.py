import streamlit as st
import onnxruntime
import cv2
import numpy as np

from PIL import Image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


st.set_page_config(
    page_title="Gradient sampling deep learning: Demo",
    page_icon="🕹️",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Demo")

# Specify the path to the saved ONNX model file
onnx_model_path = 'data/model.onnx'

# Create an ONNX runtime inference session
ort_session = onnxruntime.InferenceSession(onnx_model_path)

uploaded_photo = st.file_uploader("Choose a photo")
#img = preprocess_image(uploaded_photo)
#model, device = load_model('data/model.pth')

# Check if a photo is uploaded
if uploaded_photo is not None:
    if uploaded_photo.type not in ["image/png", "image/jpeg"]:
        st.error("This is not an image file (must be jpeg or png).")
    else:
        # Convert the uploaded file to a numpy array
        image = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.transpose(img, (0, 3, 1, 2))

        st.image(Image.open(uploaded_photo), caption="Uploaded Image.", use_column_width=True)

        if st.button('Predict'):
            # Perform inference
            output = ort_session.run(None, {'input': img.astype(np.float32)})

            # Convert the output to class label
            predicted_class = np.argmax(output)
            predicted_prob = np.around(sigmoid(output[0][0][predicted_class]) * 100.0, decimals=2)

            st.write(f"Predicted Class: **{classes[predicted_class]}**, confidence: `{predicted_prob}`%")

