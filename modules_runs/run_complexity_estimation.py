import cv2
import line_profiler

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.complexity_estimation import SceneType, Complexity, ComplexityEstimator

@line_profiler.profile
def run_complexity_estimation():
    model = ComplexityEstimator()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break
        
        complexity, weather, confidence = model.predict(SceneType.OUTDOOR, frame)

        cv2.putText(frame, f"{complexity.name}: {weather.name} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Complexity Estimation", frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_complexity_estimation()
