# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from model import detect_number_plate

st.title("Number Plate Detection with Confidence Score")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Detect with confidence score
    result_bgr = detect_number_plate(image_np)

    # Convert for Streamlit display
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    st.image(result_rgb, caption="Detected Number Plate", use_container_width=True)
