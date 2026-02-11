import cv2
import torch
import numpy as np
import line_profiler
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks

@line_profiler.profile
def run_yolo():
    # model = YOLO("weights/yolo26n.pt",       task="detect")
    # model = YOLO("weights/yolo26n.onnx",     task="detect")
    # model = YOLO("weights/yolo26n-seg.pt",   task="segment")
    model = YOLO("weights/yolo26n-seg.onnx", task="segment")
    # model.export(format="onnx")

    color = np.array([0, 255, 0], dtype=np.uint8) * 0.5
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break

        yolo_output = model(frame, verbose=True)[0]
        
        if yolo_output.masks is not None:
            masks = scale_masks(yolo_output.masks.data.unsqueeze(1), yolo_output.boxes.orig_shape, padding=True)

            for mask_model in masks:
                mask = mask_model[0].cpu().numpy() > 0.5
                frame[mask] = frame[mask] * 0.5 + color

        if yolo_output.boxes is not None:
            for box in yolo_output.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{model.names[cls]} {conf:.2f})",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO 26 Nano", frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo()
