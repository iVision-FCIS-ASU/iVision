import cv2
import line_profiler
import numpy as np
import numpy.typing as npt
from modules.depth_estimation import DepthEstimator
from modules.object_detection_tflite import ObjectDetectorTFLite
from modules.utils.depth_estimation_helpers import get_object_depth_mask
from typing import Callable, TypeAlias


Box     : TypeAlias = tuple[int, int, int, int, int, float]
Mask    : TypeAlias = np.ndarray[tuple[int, int], bool]
Centroid: TypeAlias = tuple[tuple[int, int], str]
Objects : TypeAlias = tuple[list[Box], list[Mask]]
DepthMethodType: TypeAlias = Callable[[npt.NDArray, Box], tuple[int, int, int]]

font = cv2.FONT_HERSHEY_SIMPLEX
overlay_mask_color = 0.5 * np.array((0, 255, 0), dtype=np.uint8)

@line_profiler.profile
def get_mean(depth_bw: npt.NDArray, box: Box) -> tuple[int, int, int]:
    x1, y1, x2, y2, _, _ = box
    region = depth_bw[y1:y2, x1:x2]
    min_depth = int(np.min(region))
    max_depth = int(np.max(region))
    
    depth = int(np.mean(region))
    return depth, min_depth, max_depth

@line_profiler.profile
def get_percentile_80(depth_bw: npt.NDArray, box: Box) -> tuple[int, int, int]:
    x1, y1, x2, y2, _, _ = box
    region = depth_bw[y1:y2, x1:x2]
    min_depth = int(np.min(region))
    max_depth = int(np.max(region))

    depth = int(np.percentile(region, 80))
    return depth, min_depth, max_depth

@line_profiler.profile
def get_topk_50(depth_bw: npt.NDArray, box: Box) -> tuple[int, int, int]:
    x1, y1, x2, y2, _, _ = box
    region = depth_bw[y1:y2, x1:x2].flatten()
    min_depth = int(np.min(region))
    max_depth = int(np.max(region))

    k = int(0.50 * len(region))
    top = np.partition(region, -k)[-k:]
    depth = int(np.mean(top))
    return depth, min_depth, max_depth


depth_methods: dict[int, DepthMethodType] = {
    1: get_mean,
    2: get_percentile_80,
    3: get_topk_50,
}

depth_methods_str: dict[int, str] = {
    1: "mean",
    2: "80% percentile",
    3: "top 50% mean",
}

def draw_objects(
    output_image: npt.NDArray, 
    depth_bw: npt.NDArray, 
    boxes: list[Box], 
    masks: list[Mask], 
    centroids: list[Centroid],
    depth_method: int
) -> None:

    if len(masks) == 0:
        for box, (centroid, cls_str) in zip(boxes, centroids):
            x1, y1, x2, y2, _, conf = box
            depth, min_depth, max_depth = depth_methods[depth_method](depth_bw, box)

            label = f"{cls_str}:({conf:0.2f}) depth:({depth}, {min_depth}, {max_depth})"
            cv2.circle(output_image, centroid, radius=10, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x1, y1 - 5), font, 0.5, (0, 0, 0), 5)
            cv2.putText(output_image, label, (x1, y1 - 5), font, 0.5, (0, 255, 0), 2)
    else:
        for box, mask, (centroid, cls_str) in zip(boxes, masks, centroids):
            x1, y1, x2, y2, _, conf = box
            mean_depth, min_depth, max_depth = get_object_depth_mask(depth_bw, mask)
            output_image[mask] = 0.5 * output_image[mask] + overlay_mask_color 
            
            label = f"{cls_str}:({conf:0.2f}) depth:({mean_depth}, {min_depth}, {max_depth})"
            cv2.circle(output_image, centroid, radius=10, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x1, y1 - 5), font, 0.5, (0, 0, 0), 5)
            cv2.putText(output_image, label, (x1, y1 - 5), font, 0.5, (0, 255, 0), 2)


@line_profiler.profile
def testing_depth_methods():
    model_yolo = ObjectDetectorTFLite()
    model_depth = DepthEstimator()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    model_yolo_type = ObjectDetectorTFLite.ModelType.YOLO_DETECT
    model_depth_type = DepthEstimator.ModelType.DEPTH_ANYTHING_V2
    depth_method = 1

    depth_keys = { "1", "2", "3" }
    yolo_keys = {
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

        boxes, masks, centroids = model_yolo.get_objects(frame, model_yolo_type)
        depth_bw, _ = model_depth.get_depth_image(frame, model_depth_type)
        draw_objects(output_image, depth_bw, boxes, masks, centroids, depth_method)        

        label = depth_methods_str[depth_method]
        cv2.putText(output_image, label, (20, 40), font, 1, (0, 0, 0), 5)
        cv2.putText(output_image, label, (20, 40), font, 1, (0, 255, 0), 2)
        cv2.imshow("Testing Depth Methods", output_image)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        elif key_pressed != -1 and chr(key_pressed) in depth_keys:
            depth_method = int(chr(key_pressed))
        elif key_pressed != -1 and chr(key_pressed) in yolo_keys:
            model_yolo_type = ObjectDetectorTFLite.ModelType(yolo_keys[chr(key_pressed)])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    testing_depth_methods()
