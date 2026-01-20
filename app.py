# ================= ENV FIXES =================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ================= IMPORTS =================
import streamlit as st
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ================= PAGE CONFIG (FIRST LINE!) =================
st.set_page_config(
    page_title="Sign Language Detector (A–Z)",
    page_icon="🖐️",
    layout="centered"
)

# ================= CONSTANTS =================
IMG_SIZE = 224
OFFSET = 20

# ================= LOAD MODEL (SAFE CACHE) =================
@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    with open("Model/labels.txt") as f:
        labels = [l.strip() for l in f if l.strip()]
    return model, labels

model, labels = load_model_and_labels()

# ================= CREATE MEDIAPIPE HANDS (SAFE) =================
def create_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

# ================= SAFE IMAGE READER =================
def read_image(raw_bytes):
    import cv2  # 🔥 lazy import
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img

# ================= UI HEADER =================
st.markdown(
    "<h2 style='text-align:center'>🖐️ Sign Language Detector (A–Z)</h2>",
    unsafe_allow_html=True
)

mode = st.radio(
    "Choose mode",
    ["📸 Upload Image", "🎥 Live Camera"],
    horizontal=True
)

# ================= IMAGE MODE =================
if mode == "📸 Upload Image":
    uploaded = st.file_uploader("Upload hand image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = read_image(uploaded.read())
        if img is None:
            st.error("Invalid image")
            st.stop()

        import cv2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hands = create_hands()
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            st.warning("❌ No hand detected")
            st.image(img_rgb, channels="RGB")
        else:
            lm = results.multi_hand_landmarks[0]
            h, w, _ = img.shape

            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]

            x1, y1 = max(min(xs) - OFFSET, 0), max(min(ys) - OFFSET, 0)
            x2, y2 = min(max(xs) + OFFSET, w), min(max(ys) + OFFSET, h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                st.error("Crop failed")
                st.stop()

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
            pred = model.predict(crop[np.newaxis, ...], verbose=0)
            label = labels[np.argmax(pred)]

            mp.solutions.drawing_utils.draw_landmarks(
                img_rgb, lm, mp.solutions.hands.HAND_CONNECTIONS
            )

            st.success(f"Prediction: **{label}**")
            st.image(img_rgb, channels="RGB")

        hands.close()

# ================= LIVE CAMERA MODE =================
else:
    cam = st.camera_input("Take a photo")

    if cam:
        img = read_image(cam.read())
        import cv2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hands = create_hands()
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            h, w, _ = img.shape

            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]

            x1, y1 = max(min(xs) - OFFSET, 0), max(min(ys) - OFFSET, 0)
            x2, y2 = min(max(xs) + OFFSET, w), min(max(ys) + OFFSET, h)

            crop = img[y1:y2, x1:x2]
            if crop.size:
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
                pred = model.predict(crop[np.newaxis, ...], verbose=0)
                label = labels[np.argmax(pred)]
                st.success(f"Prediction: **{label}**")

            mp.solutions.drawing_utils.draw_landmarks(
                img_rgb, lm, mp.solutions.hands.HAND_CONNECTIONS
            )
        else:
            st.warning("❌ No hand detected")

        st.image(img_rgb, channels="RGB")
        hands.close()
