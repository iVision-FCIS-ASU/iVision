from __future__ import annotations

import cv2
import line_profiler
import numpy as np
import numpy.typing as npt

from enum import Enum
from scipy import ndimage
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks
from .utils.depth_estimation_helpers import get_object_depth_box, get_object_depth_mask

class ObjectDetector:
    class ModelType(Enum):
        YOLO_DETECT = 1
        YOLO_SEGMENT = 2

    def __init__(
        self, 
        overlay_ratio: float = 0.5,
        overlay_color: tuple[int, int, int] = (0, 255, 0),
        overlay_thickness: int = 2,
        overlay_font_scale: float = 0.5,
        overlay_centroid_radius: int = 10,
    ):
        self.models = {
            self.ModelType.YOLO_DETECT: YOLO("weights/yolo26n.onnx", task="detect"),
            self.ModelType.YOLO_SEGMENT: YOLO("weights/yolo26n-seg.onnx", task="segment")
        }

        self.classes = {
            self.ModelType.YOLO_DETECT: self.models[self.ModelType.YOLO_DETECT].names,
            self.ModelType.YOLO_SEGMENT: self.models[self.ModelType.YOLO_SEGMENT].names
        }

        self.model_types = { model_type.value for model_type in ObjectDetector.ModelType }

        self.model_type: None | ObjectDetector.ModelType = None
        self.boxes: list[tuple[int, int, int, int, int, float]] = []
        self.masks: list[npt.NDArray] = []
        self.centroids: list[tuple[tuple[int, int], str]] = []

        self.overlay_color = overlay_color
        self.overlay_mask_ratio = overlay_ratio
        self.overlay_mask_inverse_ratio = 1 - overlay_ratio
        self.overlay_mask_color = overlay_ratio * np.array(self.overlay_color, dtype=np.uint8)
        self.overlay_thickness = overlay_thickness
        self.overlay_font_scale = overlay_font_scale
        self.overlay_centroid_radius = overlay_centroid_radius
    
    # @line_profiler.profile
    def get_objects(
        self, 
        frame: npt.NDArray, 
        model_yolo_type: ObjectDetector.ModelType,
        imgsz: int = 320
    ) -> tuple[list[tuple[int, int, int, int, int, float]], list[npt.NDArray], list[tuple[tuple[int, int], str]]]:
        self.model_type = model_yolo_type
        output = self.models[self.model_type].predict(frame, verbose=False, imgsz=imgsz)[0]

        self.boxes = []
        if output.boxes is not None:
            for box in output.boxes:
                x1, y1, x2, y2 = box.xyxy[0].round().int().tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                self.boxes.append((x1, y1, x2, y2, cls, conf))
        
        self.masks = []
        self.centroids = []
        if model_yolo_type != self.ModelType.YOLO_SEGMENT or output.masks is None:
            for box in self.boxes:
                x1, y1, x2, y2, cls, _ = box
                centroid = (int((x2 + x1) / 2), int((y2 + y1) / 2))
                self.centroids.append((centroid, self.classes[self.model_type][cls]))
        else:
            mask_models = scale_masks(output.masks.data.unsqueeze(1), output.boxes.orig_shape, padding=True)
            self.masks = [mask_model[0].cpu().numpy() > 0.5 for mask_model in mask_models]
            self.centroids = [
                (tuple(map(round, ndimage.center_of_mass(mask)[::-1])), self.classes[self.model_type][box[4]]) 
                for mask, box in zip(self.masks, self.boxes)
            ]
        
        return self.boxes, self.masks, self.centroids

    def draw_objects(
        self,
        output_image: npt.NDArray,
    ):
        if self.model_type is None or len(self.boxes) == 0:
            return

        if self.model_type == self.ModelType.YOLO_DETECT or len(self.masks) == 0:
            for box, (centroid, _) in zip(self.boxes, self.centroids):
                x1, y1, x2, y2, cls, conf = box
                
                label = f"{self.classes[self.model_type][cls]} {conf:0.2f}"
                cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.overlay_color, self.overlay_thickness)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.overlay_font_scale, self.overlay_color, self.overlay_thickness)
        else:
            for box, mask, (centroid, _) in zip(self.boxes, self.masks, self.centroids):
                x1, y1, x2, y2, cls, conf = box
                output_image[mask] = self.overlay_mask_inverse_ratio * output_image[mask] + self.overlay_mask_color 
                
                label = f"{self.classes[self.model_type][cls]} {conf:0.2f}"
                cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.overlay_color, self.overlay_thickness)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.overlay_font_scale, self.overlay_color, self.overlay_thickness)

    def draw_objects_with_depth(
        self,
        output_image: npt.NDArray,
        depth_bw: npt.NDArray,
        draw_masks: bool = False,
        depth_warning_threshold: int = 150
    ) -> list[tuple[tuple[int, int], str]]:
        if self.model_type is None or len(self.boxes) == 0:
            return []

        warning_centroids: list[tuple[tuple[int, int], str]] = []

        if self.model_type == self.ModelType.YOLO_DETECT or len(self.masks) == 0:
            for box, (centroid, cls_str) in zip(self.boxes, self.centroids):
                x1, y1, x2, y2, _, conf = box
                object_depth, min_depth, max_depth = get_object_depth_box(depth_bw, box)
                if object_depth >= depth_warning_threshold:
                    warning_centroids.append((centroid, cls_str))

                label = f"{cls_str}:({conf:0.2f}) depth:({object_depth}, {min_depth}, {max_depth})"
                cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.overlay_color, self.overlay_thickness)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.overlay_font_scale, self.overlay_color, self.overlay_thickness)
        else:
            for box, mask, (centroid, cls_str) in zip(self.boxes, self.masks, self.centroids):
                x1, y1, x2, y2, _, conf = box
                object_depth, min_depth, max_depth = get_object_depth_mask(depth_bw, mask)
                if object_depth >= depth_warning_threshold:
                    warning_centroids.append((centroid, cls_str))
                if draw_masks:
                    output_image[mask] = self.overlay_mask_inverse_ratio * output_image[mask] + self.overlay_mask_color 
                
                label = f"{cls_str}:({conf:0.2f}) depth:({object_depth}, {min_depth}, {max_depth})"
                cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.overlay_color, self.overlay_thickness)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.overlay_font_scale, self.overlay_color, self.overlay_thickness)
        
        return warning_centroids
        