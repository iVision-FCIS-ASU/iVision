import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("weights/complexity_estimation_v1.keras")
classes= ["cloudy","foggy","night","rainy","snowy","sunny"]

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype(np.uint8)
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict(frame):
    input_img = preprocess(frame)
    
    preds = model.predict(input_img, verbose=0)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_id]
    label = classes[class_id]

    return label, confidence

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from IP camera")
        continue

    label, confidence = predict(frame)

    cv2.putText(frame, f"{label} {confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("MobileNet IP Camera Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()