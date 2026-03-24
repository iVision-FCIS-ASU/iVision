import cv2
import line_profiler

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.object_detection import ObjectDetector
from modules.object_detection_tflite import ObjectDetectorTFLite

@line_profiler.profile
def run_object_detection():
    model_onnx = ObjectDetector() 
    model_tflite = ObjectDetectorTFLite()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    current_model = "1"
    model_types = ["1", "2"]

    current_yolo_type = 2
    yolo_types = {
        "w": 1,
        "W": 1,
        "s": 2,
        "S": 2,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break
        
        output_image = frame.copy()
        
        if current_model == "1":
            model_tflite.get_objects(frame, ObjectDetectorTFLite.ModelType(current_yolo_type))
            model_tflite.draw_objects(output_image)
        else:
            model_onnx.get_objects(frame, ObjectDetector.ModelType(current_yolo_type))
            model_onnx.draw_objects(output_image)
        
        cv2.imshow("YOLO 26 Nano", output_image)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        elif key_pressed != -1 and chr(key_pressed) in model_types:
            current_model = chr(key_pressed)
        elif key_pressed != -1 and chr(key_pressed) in yolo_types:
            current_yolo_type = yolo_types[chr(key_pressed)]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_object_detection()
