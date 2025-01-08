import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained model and face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    st.title('Emotion Detection App')
    st.write('This app detects emotions in real-time using your webcam.')
    
    # Capture image from camera
    img_file_buffer = st.camera_input("Capture Emotion")
    
    if img_file_buffer is not None:
        try:
            # Convert the image buffer to a numpy array
            image = np.array(img_file_buffer)
            
            # Convert RGB to BGR
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect emotion in the current frame
            frame = detect_emotion(frame)

            # Display the frame with emotion detection
            st.image(frame, channels='BGR', caption='Emotion Detector')
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
