from __future__ import annotations
import cv2
import torch
import numpy as np
from typing import Literal
from numpy import typing as npt
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks
from .utils.depth_estimation_helpers import get_mean_depth_box, get_mean_depth_mask
from enum import Enum
from scipy import ndimage

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

        self.model_type: None | ObjectDetector.ModelType = None
        self.boxes: None | list[tuple[int, int, int, int, int, float]] = None
        self.masks: None | list[npt.NDArray] = None
        self.centroids: None | list[tuple[int, int]] = None

        self.overlay_color = overlay_color
        self.overlay_mask_ratio = overlay_ratio
        self.overlay_mask_inverse_ratio = 1 - overlay_ratio
        self.overlay_mask_color = overlay_ratio * np.array(self.overlay_color, dtype=np.uint8)
        self.overlay_thickness = overlay_thickness
        self.overlay_font_scale = overlay_font_scale
        self.overlay_centroid_radius = overlay_centroid_radius
    
    def get_objects(
        self, 
        frame: npt.NDArray, 
        model_yolo_type: ObjectDetector.ModelType
    ) -> tuple[None | list[tuple[int, int, int, int, int, float]], None | list[npt.NDArray], list[tuple[int, int]]]:
        self.model_type = model_yolo_type
        output = self.models[self.model_type](frame, verbose=False)[0]

        if output.boxes is None:
            self.boxes = None
        else:
            self.boxes = []
            for box in output.boxes:
                x1, y1, x2, y2 = box.xyxy[0].round().int().tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                self.boxes.append((x1, y1, x2, y2, cls, conf))
        
        if model_yolo_type != self.ModelType.YOLO_SEGMENT or output.masks is None:
            self.masks = None
            self.centroids = None
            if self.boxes is not None:
                self.centroids = []
                for box in self.boxes:
                    x1, y1, x2, y2, _, _ = box
                    centroid = (int((x2 + x1) / 2), int((y2 + y1) / 2))
                    self.centroids.append(centroid)
        else:
            mask_models = scale_masks(output.masks.data.unsqueeze(1), output.boxes.orig_shape, padding=True)
            self.masks = [mask_model[0].cpu().numpy() > 0.5 for mask_model in mask_models]
            self.centroids = [tuple(map(round, ndimage.center_of_mass(mask)[::-1])) for mask in self.masks]
        
        return self.boxes, self.masks, self.centroids

    def draw_objects(
        self,
        output_image: npt.NDArray,
    ):
        if self.model_type is None or self.boxes is None:
            return

        if self.model_type == self.ModelType.YOLO_DETECT or self.masks is None:
            for box, centroid in zip(self.boxes, self.centroids):
                x1, y1, x2, y2, cls, conf = box
                # centroid = (int((x2 + x1) / 2), int((y2 + y1) / 2))
                
                label = f"{self.classes[self.model_type][cls]} {conf:0.2f}"
                cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.overlay_color, self.overlay_thickness)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.overlay_font_scale, self.overlay_color, self.overlay_thickness)
        else:
            for box, mask, centroid in zip(self.boxes, self.masks, self.centroids):
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
        draw_masks: bool = False
    ):
        if self.model_type is None or self.boxes is None:
            return

        if self.model_type == self.ModelType.YOLO_DETECT or self.masks is None:
            for box, centroid in zip(self.boxes, self.centroids):
                x1, y1, x2, y2, cls, conf = box
                # centroid = (int((x2 + x1) / 2), int((y2 + y1) / 2))
                mean_depth, min_depth, max_depth = get_mean_depth_box(depth_bw, box)

                label = f"{self.classes[self.model_type][cls]}:({conf:0.2f}) depth:({mean_depth}, {min_depth}, {max_depth})"
                cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.overlay_color, self.overlay_thickness)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.overlay_font_scale, self.overlay_color, self.overlay_thickness)
        else:
            for box, mask, centroid in zip(self.boxes, self.masks, self.centroids):
                x1, y1, x2, y2, cls, conf = box
                mean_depth, min_depth, max_depth = get_mean_depth_mask(depth_bw, mask)
                if draw_masks:
                    output_image[mask] = self.overlay_mask_inverse_ratio * output_image[mask] + self.overlay_mask_color 
                
                label = f"{self.classes[self.model_type][cls]}:({conf:0.2f}) depth:({mean_depth}, {min_depth}, {max_depth})"
                cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.overlay_color, self.overlay_thickness)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.overlay_font_scale, self.overlay_color, self.overlay_thickness)
        