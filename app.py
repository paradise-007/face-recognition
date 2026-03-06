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

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.h5')
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_face_detector():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    # Download the mediapipe face detector model
    model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    model_path = "/tmp/face_detector.tflite"

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    detector = mp_vision.FaceDetector.create_from_options(options)
    return detector

try:
    classifier = load_model()
    face_detector = load_face_detector()
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

# ── Detection ──────────────────────────────────────────────────────────────────
def detect_emotion(pil_image):
    import mediapipe as mp

    img_rgb = pil_image.convert("RGB")
    np_img = np.array(img_rgb)
    h, w = np_img.shape[:2]
    draw = ImageDraw.Draw(img_rgb)
    detected = []

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
    results = face_detector.detect(mp_image)

    if not results.detections:
        draw.text((30, 30), "No face detected", fill=(255, 0, 0))
        return img_rgb, detected

    for detection in results.detections:
        bbox = detection.bounding_box
        x1 = max(bbox.origin_x, 0)
        y1 = max(bbox.origin_y, 0)
        x2 = min(x1 + bbox.width, w)
        y2 = min(y1 + bbox.height, h)

        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=3)

        face_crop = img_rgb.crop((x1, y1, x2, y2)).convert("L").resize((48, 48))
        roi = np.array(face_crop).astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi, verbose=0)[0]
        label      = EMOTION_LABELS[prediction.argmax()]
        confidence = float(prediction.max()) * 100
        detected.append((label, confidence))

        text = f"{EMOTION_EMOJI[label]} {label} ({confidence:.1f}%)"
        draw.text((x1, max(y1 - 25, 0)), text, fill=(0, 230, 0))

    return img_rgb, detected


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
