import cv2
import numpy as np
import line_profiler

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.object_detection_tflite import ObjectDetectorTFLite
from modules.depth_estimation import DepthEstimator
from modules.grid_obstacle_detection import GridObstacleDetector
from modules.grid_warning_system import GridWarningSystem

@line_profiler.profile
def run_grid_warning_system():
    yolo_model = ObjectDetectorTFLite()
    depth_model = DepthEstimator()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    depth_warning_threshold = 180

    grid_obstacle_detector = GridObstacleDetector(w, h, depth_warning_threshold=depth_warning_threshold)
    grid_warning_system = GridWarningSystem(w, h)
    
    print(f"\nresolution (w, h): ({w}, {h})")
    print(f"grid_obstacle_detector size (w, h): {grid_obstacle_detector.grid_size}")
    print(f"grid_warning_system size (w, h): {grid_warning_system.grid_size}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break

        depth_bw, _ = depth_model.get_depth_image(frame, DepthEstimator.ModelType.DEPTH_ANYTHING_V2)
        warning_image = frame.copy()
        output_image = cv2.cvtColor(depth_bw, cv2.COLOR_GRAY2BGR)
        
        _, masks, _ = yolo_model.get_objects(frame, ObjectDetectorTFLite.ModelType.YOLO_SEGMENT)
        # yolo_centroids = yolo_model.get_warning_centroids(depth_bw, depth_warning_threshold)
        yolo_centroids = yolo_model.draw_objects_with_depth(output_image, depth_bw, draw_masks=True, 
                                                         depth_warning_threshold=depth_warning_threshold)

        grid_obstacle_detector.draw_grid(output_image, depth_bw, masks, 
                                         depth_warning_threshold, draw_gridlines=True)
        depth_centroids = grid_obstacle_detector.get_obstacle_centroids()

        grid_warning_system.draw_grid(warning_image, frame, yolo_centroids, depth_centroids)
        warnings = grid_warning_system.get_warnings()
    
        output_image = np.hstack((warning_image, output_image))
        label = f"Warning Threshold: {depth_warning_threshold}"
        cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Warning Test, Output shape: {output_image.shape}", output_image)

        print(f"\nYOLO centroids (x, y): {yolo_centroids}")
        print(f"Obstacle centroids (x, y): {depth_centroids}")
        print(f"Grid Warning Data: {grid_warning_system.grid_warning_data}")
        print(f"Grid Warnings: {warnings}\n")

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord("q") or pressed_key == ord("Q"):
            break
        if pressed_key == ord("w") or pressed_key == ord("W"):
            depth_warning_threshold = min(255, depth_warning_threshold + 5)
        if pressed_key == ord("s") or pressed_key == ord("S"):
            depth_warning_threshold = max(0, depth_warning_threshold - 5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_grid_warning_system()
