import cv2
from cv2_enumerate_cameras import enumerate_cameras

print("-----ALL AVAILABLE CAMERAS-----")
for camera_info in enumerate_cameras(cv2.CAP_DSHOW):
    print(camera_info)
print("-------------------------------")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame!")
        break

    cv2.imshow("Camera IDs Test", frame)

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord("q") or key_pressed == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()
