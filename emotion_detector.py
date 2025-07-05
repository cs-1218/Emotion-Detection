import tensorflow
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from time import sleep
import cv2
import numpy as np
import os

# Load the model and face detector
model = load_model(os.path.join(os.getcwd(), 'model_emotion.keras'))
face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml'))

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces detected, show message
    if len(faces) == 0:
        cv2.putText(frame, "No Face Found", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
    else:
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

            # Extract, resize, normalize ROI for model
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_normalized = roi_resized.astype('float') / 255.0
            roi_array = img_to_array(roi_normalized)
            roi_expanded = np.expand_dims(roi_array, axis=0)

            # Predict emotion
            preds = model.predict(roi_expanded)[0]
            label = emotion_labels[np.argmax(preds)]

            # Put label text above the face rectangle
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    # Show the frame
    cv2.imshow('Emotion Detector', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()