# ================= PAGE CONFIG (MUST BE FIRST) =================
import streamlit as st

st.set_page_config(
    page_title="Sign Language Detector (A–Z)",
    layout="centered",
    page_icon="🖐️"
)

# ================= ENV FIXES =================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ================= IMPORTS =================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ================= CONSTANTS =================
IMG_SIZE = 224
OFFSET = 20

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    with open("Model/labels.txt") as f:
        labels = [l.strip() for l in f if l.strip()]
    return model, labels

model, labels = load_model_and_labels()

# ================= HAND DETECTION FUNCTION =================
def detect_and_predict(img):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=True,   # ✅ REQUIRED
        max_num_hands=1,
        model_complexity=0,       # Cloud-safe
        min_detection_confidence=0.2
    )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, img_rgb

    lm = results.multi_hand_landmarks[0]
    h, w, _ = img.shape

    xs = [int(p.x * w) for p in lm.landmark]
    ys = [int(p.y * h) for p in lm.landmark]

    x1 = max(min(xs) - OFFSET, 0)
    y1 = max(min(ys) - OFFSET, 0)
    x2 = min(max(xs) + OFFSET, w)
    y2 = min(max(ys) + OFFSET, h)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, img_rgb

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    inp = crop / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred = model.predict(inp, verbose=0)
    label = labels[np.argmax(pred)]

    mp_draw.draw_landmarks(
        img_rgb, lm, mp_hands.HAND_CONNECTIONS
    )

    return label, img_rgb

# ================= UI =================
st.markdown("<h2 style='text-align:center'>🖐️ Sign Language Detector (A–Z)</h2>", unsafe_allow_html=True)

mode = st.radio(
    "Choose Mode",
    ["📸 Upload Image", "🎥 Live Camera"],
    horizontal=True
)

# ================= UPLOAD MODE =================
if mode == "📸 Upload Image":
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if img_file:
        img = cv2.imdecode(
            np.frombuffer(img_file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        label, output = detect_and_predict(img)

        if label is None:
            st.warning("❌ No hand detected")
        else:
            st.success(f"Prediction: **{label}**")

        st.image(output, channels="RGB")

# ================= CAMERA MODE =================
else:
    frame = st.camera_input("Take a photo")

    if frame:
        img = cv2.imdecode(
            np.frombuffer(frame.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        label, output = detect_and_predict(img)

        if label is None:
            st.warning("❌ No hand detected")
        else:
            st.success(f"Prediction: **{label}**")

        st.image(output, channels="RGB")
