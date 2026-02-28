import cv2
import line_profiler

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.object_detection import ObjectDetector

@line_profiler.profile
def run_object_detection():
    model = ObjectDetector()

    model_type = ObjectDetector.ModelType.YOLO_SEGMENT
    model_types = {
        "w": ObjectDetector.ModelType.YOLO_DETECT,
        "W": ObjectDetector.ModelType.YOLO_DETECT,
        "s": ObjectDetector.ModelType.YOLO_SEGMENT,
        "S": ObjectDetector.ModelType.YOLO_SEGMENT,
    }

    selected_size = 320
    sizes = {
        "1": 640,
        "2": 512,
        "3": 416,
        "4": 320,
    }

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break
        
        model.get_objects(frame, model_type, selected_size)
        model.draw_objects(frame)
        cv2.imshow("YOLO 26 Nano", frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        elif key_pressed != -1 and chr(key_pressed) in sizes:
            selected_size = sizes[chr(key_pressed)]
        elif key_pressed != -1 and chr(key_pressed) in model_types:
            model_type = model_types[chr(key_pressed)]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_object_detection()
