# ================= ENV FIXES (VERY IMPORTANT) =================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ================= IMPORTS =================
import streamlit as st
import numpy as np
import cv2
import math
import mediapipe as mp
import tensorflow as tf

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Sign Language Detector (A–Z)",
    layout="wide",
    page_icon="🖐️"
)

# ================= CONSTANTS =================
IMG_SIZE = 224
OFFSET = 20

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    with open("Model/labels.txt", "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    return model, labels

model, labels = load_model_and_labels()

# ================= MEDIAPIPE HANDS (CPU SAFE) =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,          # ✅ IMPORTANT
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3,     # ✅ LOW CONF → BETTER DETECTION
    min_tracking_confidence=0.3
)

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Options")
st.sidebar.markdown("""
**Instructions**
- Use good lighting  
- Plain background  
- Keep full hand visible  
""")

img_source = st.sidebar.radio(
    "Choose input method",
    ("Camera (take photo)", "Upload image"),
    index=1
)

# ================= TITLE =================
st.markdown(
    """
    <div style="text-align:center;background:#6C63FF;padding:10px;border-radius:10px">
        <h1 style="color:white">🖐️ Sign Language Detector (A–Z)</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Input Image")
    img_file = None
    if img_source == "Camera (take photo)":
        img_file = st.camera_input("Take a photo")
    else:
        img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("🔍 Result")
    result_box = st.empty()
    image_box = st.empty()

# ================= PROCESS =================
if img_file is not None:

    # Read image
    img_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image.")
        st.stop()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False

    results = hands.process(img_rgb)
    img_rgb.flags.writeable = True

    if not results.multi_hand_landmarks:
        result_box.warning("❌ No hand detected. Try better lighting / angle.")
        image_box.image(img_rgb, caption="Input Image", channels="RGB")

    else:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw skeleton
        mp_draw.draw_landmarks(
            img_rgb,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Bounding box
        h, w, _ = img.shape
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x1 = max(min(xs) - OFFSET, 0)
        y1 = max(min(ys) - OFFSET, 0)
        x2 = min(max(xs) + OFFSET, w)
        y2 = min(max(ys) + OFFSET, h)

        img_crop = img[y1:y2, x1:x2]

        if img_crop.size == 0:
            result_box.error("Crop failed. Try again.")
            st.stop()

        # Resize for model
        img_resize = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
        img_input = img_resize / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Predict
        prediction = model.predict(img_input, verbose=0)
        index = np.argmax(prediction)
        label = labels[index]

        # UI Output
        result_box.markdown(
            f"""
            <div style="text-align:center;background:#FFD700;padding:10px;border-radius:10px">
                <h2>Predicted Sign: <b>{label}</b></h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show images
        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="Skeleton Detection", channels="RGB")
        c2.image(img_resize, caption="Model Input (224×224)", channels="RGB")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    """
    💡 **Tips**
    - Avoid blur & shadows  
    - Hand should face camera  
    - Keep fingers visible  
    """
)
