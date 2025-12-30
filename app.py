import streamlit as st
from PIL import Image
import numpy as np
from model import detect_number_plate

st.title("Number Plate Detection with Confidence Score (No OpenCV)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    result_img = detect_number_plate(image_np)

    st.image(result_img, caption="Detected Number Plate", use_container_width=True)
