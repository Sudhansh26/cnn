import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
import cv2

import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Sign Language Detector",
    layout="wide",
    page_icon="🖐️"
)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Options")
st.sidebar.markdown("""
**Instructions**
- Use good lighting
- Plain background
- Center your hand
""")

img_source = st.sidebar.radio(
    "Choose input method",
    ("Camera (take photo)", "Upload image"),
    index=0
)

# ---------------- Title ----------------
st.markdown("""
<div style="text-align:center; background:#6C63FF; padding:10px; border-radius:10px">
<h1 style="color:white;">🖐️ Sign Language Detector (A–Z)</h1>
</div>
""", unsafe_allow_html=True)

# ---------------- Constants ----------------
IMG_SIZE = 300
OFFSET = 20

# ---------------- Load Model ----------------
@st.cache_resource
def load_resources():
    model_path = "Model/keras_model.h5"
    labels_path = "Model/labels.txt"

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None, None

    detector = HandDetector(maxHands=1)
    classifier = Classifier(model_path, labels_path)

    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    return detector, classifier, labels

detector, classifier, labels = load_resources()

if detector is None:
    st.error("❌ Model files not found. Add keras_model.h5 & labels.txt inside Model/")
    st.stop()

# ---------------- Layout ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Input")
    img_file = None

    if img_source == "Camera (take photo)":
        img_file = st.camera_input("Take a photo")
    else:
        img_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

with col2:
    st.subheader("🔍 Result")
    result_placeholder = st.empty()
    image_placeholder = st.empty()

# ---------------- Processing ----------------
if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image file")
        st.stop()

    hands, _ = detector.findHands(img)

    if not hands:
        result_placeholder.info("No hand detected")
        image_placeholder.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Input Image"
        )
    else:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        h_img, w_img = img.shape[:2]
        x1 = max(0, x - OFFSET)
        y1 = max(0, y - OFFSET)
        x2 = min(w_img, x + w + OFFSET)
        y2 = min(h_img, y + h + OFFSET)

        imgCrop = img[y1:y2, x1:x2]

        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = IMG_SIZE / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                wGap = (IMG_SIZE - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = IMG_SIZE / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                hGap = (IMG_SIZE - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

            result_placeholder.markdown(
                f"""
                <div style="background:#FFD700; padding:10px; border-radius:10px; text-align:center">
                <h2>Predicted Sign: {label}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB), caption="Cropped Hand")
            c2.image(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB), caption="Model Input")

        except Exception as e:
            st.error(f"Processing error: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
💡 **Tips**
- Plain background
- Full hand visible
- Good lighting
""")
