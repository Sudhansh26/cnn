# ================= ENV FIXES (MUST BE AT TOP) =================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
    page_title="Sign Language Detector",
    layout="wide",
    page_icon="🖐️"
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
    index=0
)

# ================= TITLE =================
st.markdown("""
<div style="text-align:center; background:#6C63FF; padding:10px; border-radius:10px">
<h1 style="color:white;">🖐️ Sign Language Detector (A–Z)</h1>
</div>
""", unsafe_allow_html=True)

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

# ================= MEDIAPIPE HANDS =================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

# ================= LAYOUT =================
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

# ================= PROCESSING =================
if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image")
        st.stop()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(img_rgb)

    if not result.multi_hand_landmarks:
        result_placeholder.info("No hand detected")
        st.image(img_rgb, caption="Input Image")
    else:
        h_img, w_img, _ = img.shape
        hand_landmarks = result.multi_hand_landmarks[0]

        x_list = [lm.x * w_img for lm in hand_landmarks.landmark]
        y_list = [lm.y * h_img for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_list)), int(max(x_list))
        y_min, y_max = int(min(y_list)), int(max(y_list))

        x1 = max(0, x_min - OFFSET)
        y1 = max(0, y_min - OFFSET)
        x2 = min(w_img, x_max + OFFSET)
        y2 = min(h_img, y_max + OFFSET)

        imgCrop = img[y1:y2, x1:x2]

        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

        h, w = imgCrop.shape[:2]
        aspectRatio = h / w

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

        imgInput = imgWhite.astype("float32") / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        prediction = model.predict(imgInput)
        index = np.argmax(prediction)
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
        c2.image(imgWhite, caption="Model Input (300×300)")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
💡 **Tips**
- Plain background
- Keep fingers visible
- Good lighting improves accuracy
""")
