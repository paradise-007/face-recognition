# face-recognition
face expression detection
Here’s an updated **README** file for your project that integrates both Flask and Streamlit components for emotion detection:

---

# Emotion Detection System using Flask and Streamlit

## Overview

This project implements a real-time emotion detection system that uses both Flask and Streamlit for different purposes. Flask serves as the backend, allowing the execution of a Python script for facial emotion detection. Streamlit is used for building an interactive frontend, where users can use their webcam to capture real-time images for emotion recognition.

The system uses a pre-trained Convolutional Neural Network (CNN) to classify emotions based on facial expressions. It can detect 7 different emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

## Features

- **Real-time Emotion Detection**: The system detects emotions in real-time using the webcam.
- **Flask Integration**: Flask runs a Python script to process emotion detection in the backend.
- **Streamlit Interface**: The frontend is built with Streamlit, where users can interact with the emotion detection system and see the results in real-time.
- **Face Detection with OpenCV**: OpenCV's Haar Cascade Classifier is used to detect faces in the video stream.
- **Emotion Classification with CNN**: The Keras-trained CNN model classifies emotions based on detected facial expressions.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- `Flask`
- `Streamlit`
- `opencv-python`
- `numpy`
- `keras`
- `tensorflow`

Install the required libraries using pip:

```bash
pip install Flask streamlit opencv-python numpy keras tensorflow
```

## Setup

1. **Download or Clone the Repository**:  
   Clone or download the project repository to your local machine.

2. **Pre-trained Model**:  
   The model used for emotion classification (`model.h5`) should be placed in the project directory. You can train a model or use an existing one.

3. **Haar Cascade Classifier**:  
   The Haar Cascade XML file (`haarcascade_frontalface_default.xml`) should also be present in the project folder for face detection.

4. **Webcam Access**:  
   The project requires access to the webcam for real-time emotion detection.

5. **Run the Flask Application**:
   To start the Flask backend and the webpage, run the following command:

   ```bash
   python app.py
   ```

   This will launch the Flask web application and display the necessary pages.

6. **Run Streamlit App**:
   In another terminal window, run the Streamlit app to start the emotion detection UI:

   ```bash
   streamlit run streamlit_app.py
   ```

   This will open a browser window where users can interact with the emotion detection system.

## Code Explanation

### **Flask Backend (`app.py`)**

1. **Flask Routes**:
   - `/login-signup`: Serves the HTML page for user login and signup.
   - `/`: The main page where the app’s homepage is rendered.
   - `/run-script`: Runs the Python script (`main.py`) that processes the emotion detection.

2. **Subprocess Call**:
   The `/run-script` route triggers the execution of the Python script using `subprocess.run()`. This allows running external Python scripts via Flask.

   ```python
   command = 'python -u -Xutf8 "path_to_script\\main.py"'
   subprocess.run(command, shell=True)
   ```

3. **Serving HTML Files**:
   The `login_signup` route serves the HTML file for user login and registration, while the main page is served using the `index` route.

### **Streamlit Frontend (`streamlit_app.py`)**

1. **Webcam Input**:  
   The app uses `st.camera_input()` to capture real-time images from the webcam.

2. **Emotion Detection**:  
   The webcam image is processed to detect faces using OpenCV’s Haar Cascade Classifier. The detected face region is then passed to a pre-trained Keras CNN model (`model.h5`) to classify the emotion.

   ```python
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_classifier.detectMultiScale(gray, 1.3, 5)
   ```

3. **Prediction and Display**:  
   The predicted emotion is displayed on the image feed using OpenCV. The result is shown using Streamlit’s `st.image()`.

   ```python
   prediction = classifier.predict(roi)[0]
   label = emotion_labels[prediction.argmax()]
   ```

4. **Real-Time Display**:  
   The emotion is displayed as a text label over the detected face in the webcam feed.

   ```python
   cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   ```

5. **Error Handling**:  
   Streamlit handles any errors and displays them on the interface.

   ```python
   st.error(f"Error: {e}")
   ```

## Emotions Detected

The system can detect the following emotions:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

### How It Works

1. **Start the Flask Backend**:
   The Flask app serves as the backend for the project, providing routes and endpoints for interaction.

2. **Start Streamlit App**:
   Streamlit is used to create an interactive UI where users can see their webcam feed and interact with the emotion detection system.

3. **Emotion Detection**:
   The webcam captures the user’s face, and the system detects and classifies their emotion using the pre-trained CNN model.

4. **Results Displayed**:
   The detected emotion is displayed in real-time on the webcam feed.

## License

This project is licensed under the MIT License.

---

This updated README explains both the Flask and Streamlit components and how they interact for emotion detection in real-time.
