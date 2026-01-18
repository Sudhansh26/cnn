# ================= PAGE CONFIG (MUST BE FIRST) =================
import streamlit as st

st.set_page_config(
    page_title="Sign Language Detector (A–Z)",
    layout="wide",
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

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}
.card {
    background: rgba(255,255,255,0.90);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.08);
}
.title {
    text-align:center;
    font-size:42px;
    font-weight:800;
}
.subtitle {
    text-align:center;
    color:#555;
    font-size:16px;
}
.pred {
    background:#ecfeff;
    padding:20px;
    border-radius:16px;
    text-align:center;
    font-size:26px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# ================= CONSTANTS =================
IMG_SIZE = 224
OFFSET = 20

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    with open("Model/labels.txt", "r") as f:
        labels = [l.strip() for l in f if l.strip()]
    return model, labels

model, labels = load_model_and_labels()

# ================= MEDIAPIPE (STABLE CONFIG) =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,     # REQUIRED for Streamlit
    max_num_hands=1,
    model_complexity=0,         # Cloud-safe
    min_detection_confidence=0.15,
    min_tracking_confidence=0.15
)

# ================= PREPROCESS =================
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
    return img

# ================= HEADER =================
st.markdown('<div class="title">🖐️ Sign Language Detector (A–Z)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or use live camera</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ================= MODE SELECT =================
mode = st.radio(
    "Choose Mode",
    ["📷 Live Camera", "🖼 Upload Image"],
    horizontal=True
)

# ================= LAYOUT =================
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if mode == "📷 Live Camera":
        img_file = st.camera_input("Take a photo")
    else:
        img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction")
    result_box = st.empty()
    image_box = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= PROCESS IMAGE =================
if img_file is not None:

    img_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image.")
        st.stop()

    img_rgb = preprocess(img)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        result_box.info("""
        👋 **Hand not detected**

        ✔ Open your palm  
        ✔ Move slightly away  
        ✔ Face palm to camera  
        ✔ Avoid shadows  
        """)
        image_box.image(img_rgb, channels="RGB")
        st.stop()

    hand_landmarks = results.multi_hand_landmarks[0]

    mp_draw.draw_landmarks(
        img_rgb,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )

    h, w, _ = img.shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

    x1 = max(min(xs) - OFFSET, 0)
    y1 = max(min(ys) - OFFSET, 0)
    x2 = min(max(xs) + OFFSET, w)
    y2 = min(max(ys) + OFFSET, h)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        st.error("Cropping failed.")
        st.stop()

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    inp = crop / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred = model.predict(inp, verbose=0)
    label = labels[np.argmax(pred)]

    result_box.markdown(
        f'<div class="pred">Predicted Sign: <span style="color:#2563eb">{label}</span></div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    c1.image(img_rgb, caption="Hand Detection", channels="RGB")
    c2.image(crop, caption="Model Input", channels="RGB")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
💡 **Tips**
- Keep full hand visible  
- Face palm toward camera  
- Avoid blur and shadows  
""")
