# app.py
import streamlit as st
import numpy as np
import cv2
import os
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# --- Page configuration ---
st.set_page_config(
    page_title="Sign Language Detector",
    layout="wide",
    page_icon="🖐️",
)

# --- Sidebar ---
st.sidebar.title("⚙️ Options")
st.sidebar.markdown(
    """
    **Instructions:**
    - Use good lighting and a plain background.
    - Center your hand in the image.
    - Choose Camera or Upload image.
    """
)
img_source = st.sidebar.radio("Choose input method", ("Camera (take photo)", "Upload image"), index=0)

# --- App title ---
st.markdown(
    """
    <div style='text-align:center; background-color:#6C63FF; padding:10px; border-radius:10px'>
        <h1 style='color:white;'>🖐️ Sign Language Detector (A-Z)</h1>
    </div>
    """, unsafe_allow_html=True
)

IMG_SIZE = 300
OFFSET = 20

# --- Load model ---
@st.cache_resource
def load_resources():
    model_path = "Model/keras_model.h5"
    labels_path = "Model/labels.txt"
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None, None
    detector = HandDetector(maxHands=1)
    classifier = Classifier(model_path, labels_path)
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return detector, classifier, labels

detector, classifier, labels = load_resources()
if detector is None:
    st.error("Model files not found. Please add `Model/keras_model.h5` and `Model/labels.txt` to the repo.")
    st.stop()

# --- Main layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Input")
    img_file = None
    if img_source == "Camera (take photo)":
        img_file = st.camera_input("Take a photo")
    else:
        img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("🔍 Result")
    result_placeholder = st.empty()
    image_placeholder = st.empty()

# --- Processing ---
if img_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Couldn't read the image. Try another file.")
        st.stop()

    hands, img_with_drawings = detector.findHands(img)
    if not hands:
        result_placeholder.info("No hand detected. Try better lighting and centering.")
        image_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Input image")
    else:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Safe cropping
        h_img, w_img = img.shape[:2]
        x1 = max(0, x - OFFSET)
        y1 = max(0, y - OFFSET)
        x2 = min(w_img, x + w + OFFSET)
        y2 = min(h_img, y + h + OFFSET)
        imgCrop = img[y1:y2, x1:x2]

        # White background
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = IMG_SIZE / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                wGap = math.ceil((IMG_SIZE - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = IMG_SIZE / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                hGap = math.ceil((IMG_SIZE - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
        except Exception as e:
            result_placeholder.error(f"Error processing crop: {e}")
            image_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Input image")
        else:
            # Predict
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index] if index < len(labels) else f"Index {index}"
            result_placeholder.markdown(
                f"<div style='text-align:center; background-color:#FFD700; padding:10px; border-radius:10px;'>"
                f"<h2 style='color:black;'>Predicted: {label}</h2></div>", unsafe_allow_html=True
            )

            # Show images side by side
            colA, colB = st.columns(2)
            colA.image(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB), channels="RGB", caption="Cropped hand")
            colB.image(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB), channels="RGB", caption="Model input (300x300)")

# --- Footer / Tips ---
st.markdown("---")
st.markdown(
    """
    <div style='background-color:#F0F2F6; padding:10px; border-radius:10px'>
    <h3>💡 Tips:</h3>
    <ul>
        <li>Use plain background and good lighting.</li>
        <li>Center your hand in the image for best results.</li>
        <li>If your `keras_model.h5` is >100MB, consider hosting it externally.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True
)
