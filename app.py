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

st.markdown("""
<style>
/* Page background */
.main {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
}

/* Title */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

/* Card effect */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Radio buttons */
div[role="radiogroup"] > label {
    background: white;
    padding: 12px 16px;
    border-radius: 12px;
    margin-right: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

/* Success box */
.stSuccess {
    border-radius: 12px;
    font-size: 18px;
}

/* Camera box */
video {
    border-radius: 16px !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ================= PAGE CONFIG (MOBILE SAFE) =================
st.set_page_config(
    page_title="Sign Language Detector",
    layout="centered",   # 📱 Mobile-friendly
    page_icon="🖐️"
)

# ================= CONSTANTS =================
IMG_SIZE = 224
OFFSET = 20

# ================= LOAD MODEL (ONCE) =================
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    with open("Model/labels.txt") as f:
        labels = [l.strip() for l in f if l.strip()]
    return model, labels

model, labels = load_model_and_labels()

# ================= MEDIAPIPE (SESSION SAFE) =================
if "hands" not in st.session_state:
    st.session_state.hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

hands = st.session_state.hands
mp_draw = mp.solutions.drawing_utils

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

# ================= IMAGE MODE =================
if mode == "📸 Upload Image":
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if img_file:
        raw = img_file.read()
        if not raw:
            st.warning("Empty image. Try again.")
            st.stop()

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Invalid image.")
            st.stop()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            st.warning("No hand detected.")
            st.image(img_rgb, channels="RGB")
        else:
            lm = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]

            x1, y1 = max(min(xs)-OFFSET,0), max(min(ys)-OFFSET,0)
            x2, y2 = min(max(xs)+OFFSET,w), min(max(ys)+OFFSET,h)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                st.error("Crop failed.")
                st.stop()

            inp = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
            pred = model.predict(inp[np.newaxis,...], verbose=0)
            label = labels[np.argmax(pred)]

            mp_draw.draw_landmarks(img_rgb, lm, mp.solutions.hands.HAND_CONNECTIONS)

            st.success(f"Prediction: **{label}**")
            st.image(img_rgb, channels="RGB")

# ================= LIVE CAMERA MODE =================
else:
    frame = st.camera_input("Live Camera")

    if frame:
        raw = frame.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]

            x1, y1 = max(min(xs)-OFFSET,0), max(min(ys)-OFFSET,0)
            x2, y2 = min(max(xs)+OFFSET,w), min(max(ys)+OFFSET,h)
            crop = img[y1:y2, x1:x2]

            if crop.size:
                inp = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
                pred = model.predict(inp[np.newaxis,...], verbose=0)
                label = labels[np.argmax(pred)]
                st.success(f"Prediction: **{label}**")

            mp_draw.draw_landmarks(img_rgb, lm, mp.solutions.hands.HAND_CONNECTIONS)

        st.image(img_rgb, channels="RGB")
