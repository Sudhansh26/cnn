# ================= ENV FIXES =================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ================= IMPORTS =================
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Sign Language Detector",
    layout="centered",
    page_icon="🖐️"
)

# ================= CONSTANTS =================
IMG_SIZE = 224
OFFSET = 20

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    with open("Model/labels.txt") as f:
        labels = [l.strip() for l in f if l.strip()]
    return model, labels

model, labels = load_model_and_labels()

# ================= UI =================
st.markdown(
    "<h2 style='text-align:center'>🖐️ Sign Language Detector (A–Z)</h2>",
    unsafe_allow_html=True
)

mode = st.radio(
    "Choose Mode",
    ["📸 Upload Image", "🎥 Live Camera"],
    horizontal=True
)

# ================= IMAGE PROCESS FUNCTION =================
def detect_and_predict(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3
    ) as hands:

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, img_rgb

        lm = results.multi_hand_landmarks[0]
        h, w, _ = img.shape

        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]

        x1, y1 = max(min(xs)-OFFSET,0), max(min(ys)-OFFSET,0)
        x2, y2 = min(max(xs)+OFFSET,w), min(max(ys)+OFFSET,h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None, img_rgb

        inp = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(inp[np.newaxis,...], verbose=0)
        label = labels[np.argmax(pred)]

        mp.solutions.drawing_utils.draw_landmarks(
            img_rgb, lm, mp.solutions.hands.HAND_CONNECTIONS
        )

        return label, img_rgb

# ================= UPLOAD IMAGE MODE =================
if mode == "📸 Upload Image":
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if img_file:
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        label, out = detect_and_predict(img)

        if label:
            st.success(f"Prediction: **{label}**")
        else:
            st.warning("No hand detected")

        st.image(out, channels="RGB")

# ================= CAMERA MODE =================
else:
    cam = st.camera_input("Take a photo")

    if cam:
        img = cv2.imdecode(np.frombuffer(cam.read(), np.uint8), cv2.IMREAD_COLOR)
        label, out = detect_and_predict(img)

        if label:
            st.success(f"Prediction: **{label}**")
        else:
            st.warning("No hand detected")

        st.image(out, channels="RGB")
