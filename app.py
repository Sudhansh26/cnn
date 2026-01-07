import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Sign Language Detector",
    layout="centered",
    page_icon="🖐️"
)

st.title("🖐️ Sign Language Detector (A–Z)")
st.markdown("Upload a hand image to predict the sign.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model_path = "Model/keras_model.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

labels_path = "Model/labels.txt"
if not os.path.exists(labels_path):
    st.error("labels.txt not found inside Model folder")
    st.stop()

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

if model is None:
    st.error("keras_model.h5 not found inside Model folder")
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Instructions")
st.sidebar.markdown(
    """
    • Use clear hand images  
    • Plain background works best  
    • One hand only  
    • JPG / PNG format  
    """
)

# ---------------- Image Upload ----------------
uploaded = st.file_uploader(
    "📤 Upload Hand Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- Prediction ----------------
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # MUST match training size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    index = int(np.argmax(preds))
    confidence = float(preds[0][index])

    label = labels[index] if index < len(labels) else "Unknown"

    st.markdown(
        f"""
        <div style="background-color:#FFD700;padding:15px;border-radius:10px;text-align:center">
            <h2 style="color:black;">Prediction: {label}</h2>
            <h4 style="color:black;">Confidence: {confidence:.2f}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    """
    **Note:**  
    This deployed version uses image-based inference for stability.  
    Real-time webcam hand detection is supported in the local version.
    """
)
