import streamlit as st
from PIL import Image
import numpy as np

# --- Title ---
st.title("Trash Classification App (Demo)")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload a trash image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# --- Model Selection ---
model_choice = st.selectbox(
    "Choose a model:",
    ["MobileNetV2", "ResNet101V2", "ResNet152V2", "MobileNet", "MobileNetV3Small", "MobileNetV3Large"]
)

# --- Predict Button ---
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # --- Dummy Prediction Logic ---
        dummy_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        predicted_class = np.random.choice(dummy_classes)
        confidence = np.random.uniform(0.7, 0.99)

        st.success(f"Predicted Trash Type: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}**")
