import cv2
import numpy as np
import line_profiler
from typing import Literal
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks
from models_download import download_models
from depth_models import MiDaS, DepthAnythingV2
from depth_helpers import get_mean_depth_box, get_mean_depth_mask

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
            model_depth = MiDaS(model_depth_type)
        case "dpt_swin2_tiny_256":
            model_depth = MiDaS(model_depth_type)
        case "depth_anything_v2":
            model_depth = DepthAnythingV2()
        case _:
            print("ERROR: Invalid Depth Model!")
            assert False

    return model_yolo, model_depth, get_mean_method

def get_output_image(yolo_classes, r, depth_bw, depth_rgb, get_mean_method):
    if get_mean_method == get_mean_depth_mask and r.masks is not None:
        masks = scale_masks(r.masks.data.unsqueeze(1), r.boxes.orig_shape, padding=True)
    else:
        masks = np.zeros(len(r.boxes))

    for box, mask_model in zip(r.boxes, masks):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        mean_depth, min_depth, max_depth = get_mean_method(depth_bw, box, mask_model)
        label = f"{yolo_classes[cls]} {conf:.2f} Depth:({mean_depth}, {min_depth}, {max_depth})"

        cv2.rectangle(depth_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            depth_rgb,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return depth_rgb

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
            continue
        
        r = model_yolo(frame, verbose=False)[0]
        if r.boxes is None:
            continue

        depth_bw, depth_rgb = model_depth.get_depth_image(frame)
        
        output_image = get_output_image(yolo_classes, r, depth_bw, depth_rgb, get_mean_method)

        if side_by_side:
            output_image = np.hstack((frame, output_image))
        cv2.imshow(f"Object Detection + Depth Estimation", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    download_models()

    run(
        model_yolo_type="segment",
        model_depth_type="depth_anything_v2",
        side_by_side=False
    )
    