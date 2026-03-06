import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# ── Load mediapipe & model ─────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    import mediapipe as mp
    import tensorflow as tf

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.h5')
    model = tf.keras.models.load_model(model_path)

    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    return face_detection, model

try:
    face_detection, classifier = load_resources()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJI  = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲',
}

# ── Detection function ─────────────────────────────────────────────────────────
def detect_emotion(pil_image):
    import mediapipe as mp

    img_rgb = np.array(pil_image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    draw = ImageDraw.Draw(pil_image)
    detected = []

    results = face_detection.process(img_rgb)

    if not results.detections:
        draw.text((30, 30), "No face detected", fill=(255, 0, 0))
        return pil_image, detected

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = min(x1 + bw, w)
        y2 = min(y1 + bh, h)

        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=3)

        face_crop = pil_image.crop((x1, y1, x2, y2)).convert("L").resize((48, 48))
        roi = np.array(face_crop).astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi, verbose=0)[0]
        label      = EMOTION_LABELS[prediction.argmax()]
        confidence = float(prediction.max()) * 100
        detected.append((label, confidence))

        text = f"{EMOTION_EMOJI[label]} {label} ({confidence:.1f}%)"
        draw.text((x1, max(y1 - 25, 0)), text, fill=(0, 230, 0))

    return pil_image, detected


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("😊 Real-Time Emotion Detector")
st.markdown("Detect facial emotions using a CNN model — take a photo or upload an image.")
st.markdown("---")

if not model_loaded:
    st.error(f"Failed to load model: {load_error}")
    st.stop()

mode = st.radio(
    "Choose input method:",
    ["📷 Take a photo", "🖼️ Upload an image"],
    horizontal=True
)

img_buffer = (
    st.camera_input("Point your camera at a face")
    if mode == "📷 Take a photo"
    else st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
)

if img_buffer is not None:
    try:
        pil_image = Image.open(img_buffer).convert("RGB")

        with st.spinner("Analyzing emotion..."):
            result_image, emotions = detect_emotion(pil_image)

        st.image(result_image, caption="Emotion Detection Result", use_column_width=True)

        if emotions:
            st.markdown("### Detected Emotion(s)")
            for label, confidence in emotions:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"## {EMOTION_EMOJI.get(label, '')} {label}")
                with col2:
                    st.progress(int(confidence))
                    st.caption(f"Confidence: {confidence:.1f}%")
        else:
            st.warning("No face detected. Make sure your face is clearly visible and well-lit.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption("Built with Streamlit · MediaPipe · TensorFlow · CNN  |  Detects: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise")
