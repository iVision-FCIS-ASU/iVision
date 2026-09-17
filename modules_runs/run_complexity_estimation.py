import cv2
import numpy as np
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

    scene_type = SceneType.INDOOR

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break
        
        complexity, weather, confidence = model.predict(frame, scene_type)

        # frame_lum = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
        frame_lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        percentile_threshold = 85
        label_lum_info = f"(min, max, avg, med, std, {percentile_threshold}%)"
        label_lum_values = (
            f"({np.min(frame_lum):>3}, {np.max(frame_lum):>3}, "
            f"{np.mean(frame_lum):>3.0f}, {np.median(frame_lum):>3.0f}, "
            f"{np.std(frame_lum):>3.0f}, {np.percentile(frame_lum, percentile_threshold):>3.0f})"
        )

        frame_lum = cv2.cvtColor(frame_lum, cv2.COLOR_GRAY2BGR)

        cv2.putText(frame_lum, label_lum_info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame_lum, label_lum_info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_lum, label_lum_values, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame_lum, label_lum_values, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        label_complexity = f"{complexity.name}: {weather.name} ({confidence:.2f})"
        cv2.putText(frame, label_complexity, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame, label_complexity, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Complexity Estimation", (np.hstack((frame, frame_lum))))

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        elif key_pressed == ord("1") or key_pressed == ord("2"):
            scene_type = SceneType(int(chr(key_pressed)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_complexity_estimation()
