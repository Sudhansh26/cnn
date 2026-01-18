# ================= IMPORTS (ONLY IMPORTS ABOVE) =================
import os
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

# ================= PAGE CONFIG (MUST BE FIRST STREAMLIT CALL) =================
st.set_page_config(
    page_title="Sign Language Detector (A–Z)",
    page_icon="✋",
    layout="wide"
)

# ================= ENV FIXES =================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ================= CSS (REACT-LIKE UI) =================
st.markdown("""
<style>
body {
    background-color: #f4f6fb;
}
.card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    animation: fadeUp 0.6s ease;
}
@keyframes fadeUp {
    from {opacity:0; transform: translateY(20px);}
    to {opacity:1; transform: translateY(0);}
}
.title {
    text-align:center;
    font-size:40px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="title">✋ Sign Language Detector (A–Z)</div>', unsafe_allow_html=True)
st.markdown("---")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    labels = open("Model/labels.txt").read().splitlines()
    return model, labels

model, labels = load_model()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ================= UI =================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">📷 Upload Image</div>', unsafe_allow_html=True)
    img_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown('<div class="card">🔍 Prediction</div>', unsafe_allow_html=True)
    result_box = st.empty()

# ================= PROCESS =================
if img_file:
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        result_box.error("❌ No hand detected")
    else:
        h, w, _ = img.shape
        xs, ys = [], []
        for lm in result.multi_hand_landmarks[0].landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        x1, y1 = max(min(xs)-20,0), max(min(ys)-20,0)
        x2, y2 = min(max(xs)+20,w), min(max(ys)+20,h)
        crop = img[y1:y2, x1:x2]

        crop = cv2.resize(crop, (224,224)) / 255.0
        crop = np.expand_dims(crop, axis=0)

        pred = model.predict(crop, verbose=0)
        label = labels[np.argmax(pred)]

        result_box.markdown(
            f"""
            <div class="card">
                <h2 style="text-align:center">Predicted Sign</h2>
                <h1 style="text-align:center;color:#6C63FF">{label}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(img_rgb, caption="Detected Hand", use_column_width=True)
