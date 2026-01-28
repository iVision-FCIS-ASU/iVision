import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from complexity_estimation_models import Complexity, ComplexityEstimation

all_camera_idx_available = []

for camera_idx in range(10):
    cap = cv2.VideoCapture(camera_idx)
    if cap.isOpened():
        print(f'Camera index available: {camera_idx}')
        all_camera_idx_available.append(camera_idx)
        cap.release()

print(all_camera_idx_available)

# model = load_model("MainData_UnFreeze_All_converted.keras")

model = ComplexityEstimation()

# cap = cv2.VideoCapture("http://10.141.137.21:8080/video")
# cap = cv2.VideoCapture("https://10.199.158.59:8080/video")
# cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from IP camera")
        break
    
    # complexity, weather, confidence = model.predict("outdoor", frame)

    cv2.putText(frame, f"{complexity.name}: {weather} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("MobileNet IP Camera Classification", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()