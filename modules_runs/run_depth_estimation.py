import cv2
import numpy as np
import line_profiler

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.depth_estimation import DepthEstimator

@line_profiler.profile
def run_depth_estimation():
    depth_estimator = DepthEstimator()
    model_type = DepthEstimator.ModelType.DEPTH_ANYTHING_V2

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    model_switch_keys = [ord(str(index)) for index in depth_estimator.model_types]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break

        depth_bw, depth_rgb = depth_estimator.get_depth_image(frame, model_type)
        output_image = np.hstack((cv2.cvtColor(depth_bw, cv2.COLOR_GRAY2BGR), depth_rgb))
        cv2.imshow(f"Depth Estimation {output_image.shape}", output_image)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        elif key_pressed in model_switch_keys:
            model_type = DepthEstimator.ModelType(int(chr(key_pressed)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_depth_estimation()
