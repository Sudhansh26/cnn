# ================= PAGE CONFIG (MUST BE FIRST) =================
import streamlit as st

st.set_page_config(
    page_title="Sign Language Detector (A–Z)",
    page_icon="✋",
    layout="wide"
)

# ================= ENV FIXES =================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ================= IMPORTS =================
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

# ================= CUSTOM CSS (REACT-LIKE UI) =================
st.markdown("""
<style>
body { background-color:#f6f7fb; }

.title {
    text-align:center;
    font-size:44px;
    font-weight:900;
    margin-bottom:5px;
}
.subtitle {
    text-align:center;
    color:#666;
    margin-bottom:40px;
}
.card {
    background:white;
    border-radius:20px;
    padding:25px;
    box-shadow:0 15px 35px rgba(0,0,0,0.12);
}
.card:hover {
    transform: translateY(-4px);
}
.result {
    font-size:52px;
    font-weight:900;
    color:#6C63FF;
    text-align:center;
}
.mode {
    font-size:18px;
    font-weight:600;
}
.footer {
    text-align:center;
    color:#888;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="title">✋ Sign Language Detector (A–Z)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or use live camera to detect hand signs</div>', unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    with open("Model/labels.txt") as f:
        labels = [l.strip() for l in f]
    return model, labels

model, labels = load_model()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ================= MODE SELECTION =================
mode = st.radio(
    "Choose Mode",
    ["📷 Live Camera", "🖼 Upload Image"],
    horizontal=True
)

col1, col2 = st.columns([1.2, 1])

# ================= IMAGE INPUT =================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    img = None

    if mode == "📷 Live Camera":
        st.markdown('<p class="mode">Live Camera</p>', unsafe_allow_html=True)
        cam = st.camera_input("Take a photo")
        if cam:
            img = cv2.imdecode(np.frombuffer(cam.read(), np.uint8), 1)

    else:
        st.markdown('<p class="mode">Upload Image</p>', unsafe_allow_html=True)
        file = st.file_uploader("Upload JPG / PNG", type=["jpg","jpeg","png"])
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION =================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            st.warning("❌ No hand detected")
            st.image(img_rgb, channels="RGB")
        else:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img_rgb, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape
            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]

            x1, x2 = max(min(xs)-20,0), min(max(xs)+20,w)
            y1, y2 = max(min(ys)-20,0), min(max(ys)+20,h)

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224,224)) / 255.0
            crop = np.expand_dims(crop, axis=0)

            pred = model.predict(crop, verbose=0)
            label = labels[np.argmax(pred)]

            st.markdown(f'<div class="result">{label}</div>', unsafe_allow_html=True)
            st.image(img_rgb, channels="RGB", caption="Hand Detection")

    else:
        st.info("👈 Upload an image or use camera")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown('<div class="footer">Built with ❤️ using Streamlit · MediaPipe · TensorFlow</div>', unsafe_allow_html=True)
