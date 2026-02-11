import cv2
import line_profiler
from scene_classification import SceneClassifier

@line_profiler.profile
def run_scene_classification():
    model = SceneClassifier()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break

        scene_binary, scene = model.get_scene_type(frame)

        cv2.putText(frame, f"{scene_binary}: {scene}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Scene Understanding", frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_scene_classification()
