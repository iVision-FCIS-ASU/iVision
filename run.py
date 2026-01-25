import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks
import line_profiler
from depth_anything_v2 import DepthAnythingV2
from depth_helpers import get_mean_depth_box, get_mean_depth_mask
from typing import Literal

def get_models(
        model_yolo_type: Literal["detect", "segment"],
        model_depth_type: Literal["midas_v21_small_256", "dpt_swin2_tiny_256", "depth_anything_v2"]
    ):
    match model_yolo_type:
        case "detect":
            model_yolo = YOLO("weights/yolo26n.onnx", task="detect")
            get_mean_method = get_mean_depth_box
        case "segment":
            model_yolo = YOLO("weights/yolo26n-seg.onnx", task="segment")
            get_mean_method = get_mean_depth_mask
        case _:
            print("ERROR: Invalid YOLO Model!")
            assert False
    
    match model_depth_type:
        case "midas_v21_small_256":
            pass
        case "dpt_swin2_tiny_256":
            pass
        case "depth_anything_v2":
            model_depth = DepthAnythingV2()
        case _:
            print("ERROR: Invalid Depth Model!")
            assert False

    return model_yolo, model_depth, get_mean_method

def get_output_image(yolo_classes, r, depth_image, depth_bw, get_mean_method):
    if get_mean_method == get_mean_depth_mask:
        masks = scale_masks(r.masks.data.unsqueeze(1), r.boxes.orig_shape, padding=True)
    else:
        masks = np.zeros(len(r.boxes))

    for box, mask_model in zip(r.boxes, masks):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        mean_depth, min_depth, max_depth = get_mean_method(depth_bw, box, mask_model)
        label = f"{yolo_classes[cls]} {conf:.2f} Depth:({mean_depth}, {min_depth}, {max_depth})"

        cv2.rectangle(depth_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            depth_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return depth_image

def run(
        model_yolo_type: Literal["detect", "segment"],
        model_depth_type: Literal["midas_v21_small_256", "dpt_swin2_tiny_256", "depth_anything_v2"],
        side_by_side: bool = False
    ):

    cap = cv2.VideoCapture(0)
    model_yolo, model_depth, get_mean_method = get_models(model_yolo_type, model_depth_type)
    yolo_classes = model_yolo.names

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame!")
            continue
        
        r = model_yolo(frame, verbose=True)[0]
        if r.boxes is None:
            continue

        depth_bw, depth_image = model_depth.get_depth_image(frame)

        output_image = get_output_image(yolo_classes, r, depth_image, depth_bw, get_mean_method)

        if side_by_side:
            output_image = np.hstack((frame, output_image))
        cv2.imshow(f"Object Detection + Depth Estimation", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(
        model_yolo_type="detect",
        model_depth_type="depth_anything_v2",
        side_by_side=False
    )
    