import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# ── Load model & classifier (cached so they load only once) ────────────────────
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    xml_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
    model_path = os.path.join(base_dir, 'model.h5')

    face_clf = cv2.CascadeClassifier(xml_path)
    model = load_model(model_path)
    return face_clf, model

face_classifier, classifier = load_resources()

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_EMOJI = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😄',
    'Neutral':  '😐',
    'Sad':      '😢',
    'Surprise': '😲',
}

# ── Emotion detection function ─────────────────────────────────────────────────
def detect_emotion(frame_bgr):
    """Run face detection + emotion classification on a BGR numpy frame.
    Returns (annotated_frame, list_of_detected_emotions)."""
    detected = []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame_bgr, 'No face detected', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame_bgr, detected

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi, verbose=0)[0]
            label = EMOTION_LABELS[prediction.argmax()]
            confidence = float(prediction.max()) * 100
            detected.append((label, confidence))

            display_text = f"{label} ({confidence:.1f}%)"
            cv2.putText(frame_bgr, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame_bgr, detected


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("😊 Real-Time Emotion Detector")
st.markdown("Use your webcam to detect facial emotions powered by a CNN model.")
st.markdown("---")

mode = st.radio("Choose input method:", ["📷 Take a photo", "🖼️ Upload an image"], horizontal=True)

if mode == "📷 Take a photo":
    img_buffer = st.camera_input("Point your camera at a face and click the button")
else:
    img_buffer = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if img_buffer is not None:
    try:
        # Convert to numpy BGR frame
        pil_image = Image.open(img_buffer).convert("RGB")
        frame = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing emotion..."):
            result_frame, emotions = detect_emotion(frame_bgr)

        # Convert back to RGB for display
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Emotion Detection Result", use_column_width=True)

        # Show results
        if emotions:
            st.markdown("### Detected Emotion(s)")
            for label, confidence in emotions:
                emoji = EMOTION_EMOJI.get(label, "")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"## {emoji} {label}")
                with col2:
                    st.progress(int(confidence))
                    st.caption(f"Confidence: {confidence:.1f}%")
        else:
            st.warning("No face detected. Please make sure your face is clearly visible and well-lit.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption("Built with Streamlit · OpenCV · Keras · CNN  |  Detects: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise")
