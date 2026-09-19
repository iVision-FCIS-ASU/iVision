import numpy as np
import numpy.typing as npt
from typing import TypeAlias

NDArrayf32: TypeAlias = npt.NDArray[np.float32]
NDArrayi32: TypeAlias = npt.NDArray[np.int32]
NDArrayip: TypeAlias = npt.NDArray[np.intp]

def extract_filter(
    preds: NDArrayf32, 
    conf_thresh: float = 0.25
) -> tuple[NDArrayf32, NDArrayip, NDArrayf32]:
    """Extracts and filters the objects from the YOLO predictions.

    Args:
        preds: Pre-NMS predictions from the YOLO model, shape=[1, 4 + classes, n].
        conf_thresh: Confidence threshold used to filter the objects.
    
    Returns:
        Bounding boxes (raw xywh), class IDs and confidences.
        Shapes are [m, 4], [m] and [m] respectively.
    """
    preds = preds[0].T
    boxes = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids: NDArrayip = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(class_scores.shape[0]), class_ids]

    filter_mask = confs >= conf_thresh
    boxes = boxes[filter_mask, :]
    class_ids = class_ids[filter_mask]
    confs = confs[filter_mask]

    return boxes, class_ids, confs

def xywh2xyxy(
        boxes: NDArrayf32
    ) -> NDArrayf32:
    """Transforms the boxes' coordinates from xywh to xyxy.

    Args:
        boxes: Bounding boxes (xywh).
    
    Returns:
        Bounding boxes (xyxy).
    """
    half_widths = boxes[:, 2] * 0.5
    boxes[:, 2] = boxes[:, 0] + half_widths
    boxes[:, 0] = boxes[:, 0] - half_widths
    
    half_heights = boxes[:, 3] * 0.5
    boxes[:, 3] = boxes[:, 1] + half_heights
    boxes[:, 1] = boxes[:, 1] - half_heights

    return boxes

def nms(
    boxes: NDArrayf32, 
    class_ids: NDArrayip,
    confs: NDArrayf32,
    iou_thresh: float = 0.7
) -> tuple[NDArrayf32, NDArrayip, NDArrayf32]:
    """Applies class-aware NMS to filter the objects.

    Args:
        boxes: Bounding boxes (xyxy).
        class_ids: Class IDs.
        confs: Confidences.
        iou_thresh: IoU threshold used to filter duplicates.
    
    Returns:
        Filtered boxes, class IDs and confidences.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    indices = np.argsort(confs)

    keep_indices = []
    while len(indices) > 0:
        i_curr = indices[-1]
        i_rest = indices[: -1]

        keep_indices.append(i_curr)
        
        xx1 = np.maximum(x1[i_curr], x1[i_rest])
        yy1 = np.maximum(y1[i_curr], y1[i_rest])
        xx2 = np.minimum(x2[i_curr], x2[i_rest])
        yy2 = np.minimum(y2[i_curr], y2[i_rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        union = areas[i_curr] + areas[i_rest] - inter
        iou = inter / union

        mask = (iou <= iou_thresh) | (class_ids[i_curr] != class_ids[i_rest])
        indices = i_rest[mask]

    boxes = boxes[keep_indices, :]
    class_ids = class_ids[keep_indices]
    confs = confs[keep_indices]

    return boxes, class_ids, confs

def revert_coords(
    boxes: NDArrayf32,
    orig_shape: tuple[int, int],
    pad_info: tuple[int, int, int, int],
    scale: float,
    nn_size: int
) -> NDArrayi32:
    """Reverts the boxes' coordinates to the original image dimensions.

    Applies a series of transformations to the boxes' coordinates:
        1. Denormalizes from [0, 1] to [0, nn_size]
        2. Undos the padding.
        3. Undos the scaling.
        4. Clips every xy between [0, orig_w - 1] and [0, orig_h - 1] respectively.
        5. Converts coordinates datatype to np.int32.
    
    Args:
        boxes: Bounding boxes (xyxy), shape=[n, 4].
        orig_shape: Original image dimensions, expected=(h, w).
        pad_info: Padding values used in preprocessing, expected=(top, bottom, left, right).
        nn_size: Model input dimension.
        scale: Scaling value used in preprocessing.

    Returns:
        Transformed bounding boxes.
    """
    h, w = orig_shape
    h, w = h - 1, w - 1
    top, _, left, _ = pad_info

    boxes *= nn_size
    boxes -= np.array([left, top, left, top])
    boxes *= 1 / scale
    boxes = np.clip(boxes, np.zeros(4), [w, h, w, h]).astype(np.int32)

    return boxes
