from __future__ import annotations

import cv2
import line_profiler
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from enum import Enum
from scipy import ndimage
from typing import Literal, TypeAlias
from .utils.depth_estimation_helpers import get_object_depth_box, get_object_depth_mask
from .utils.object_detection_helpers import extract_filter, nms, revert_coords, xywh2xyxy

npimagebgr : TypeAlias = np.ndarray[tuple[int, int, Literal[3]], np.uint8]
npimagegray: TypeAlias = np.ndarray[tuple[int, int], np.uint8]

Box     : TypeAlias = tuple[int, int, int, int, int, float]
Mask    : TypeAlias = np.ndarray[tuple[int, int], bool]
Centroid: TypeAlias = tuple[tuple[int, int], str]
Objects : TypeAlias = tuple[list[Box], list[Mask]]

COCO_CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
                "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", 
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
                "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class YOLODetect:
    def __init__(self):
        # self.__interpreter = tf.lite.Interpreter(f"weights/yolo26n_float32.tflite", num_threads=4)
        self.__interpreter = tf.lite.Interpreter(f"weights/yolo26n_coco_f32.tflite", num_threads=4)
        self.__interpreter.allocate_tensors()
        self.__input_index = self.__interpreter.get_input_details()[0]["index"]
        self.__output_index = self.__interpreter.get_output_details()[0]["index"]

        self.__nn_size = 320
        self.classes = COCO_CLASSES

    # @line_profiler.profile
    def __preprocess(
        self,
        frame: npt.NDArray
    ) -> tuple[npt.NDArray, tuple[int, int], tuple[int, int, int, int], float]:
        h, w = frame.shape[:2]
        scale = self.__nn_size / max(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pad_h = self.__nn_size - new_h
        pad_w = self.__nn_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = cv2.multiply(img, 1.0 / 255.0, dtype=cv2.CV_32F)
        img = np.expand_dims(img, axis=0)

        return img, frame.shape[:2], (top, bottom, left, right), scale

    # @line_profiler.profile
    def __postprocess(
        self,
        preds: npt.NDArray, 
        orig_shape: tuple[int, int],
        pad_info: tuple[int, int, int, int],
        scale: float
    ) -> list[Box]:
        boxes, class_ids, confs = extract_filter(preds)

        if len(boxes) == 0:
            return []

        # check if this can be replaced with
        # cv2.dnn.NMSBoxesBatched() to produce 
        # better performance
        boxes = xywh2xyxy(boxes)
        boxes, class_ids, confs = nms(boxes, class_ids, confs)

        boxes = revert_coords(boxes, orig_shape, pad_info, scale, self.__nn_size)
        
        # creating our objects
        final_boxes = [
            (*box, class_id, conf)
            for box, class_id, conf in zip(boxes, class_ids, confs)
        ]
        
        return final_boxes

    # @line_profiler.profile
    def get_objects(self, frame: npimagebgr) -> Objects:
        image, orig_shape, pad_info, scale = self.__preprocess(frame)

        self.__interpreter.set_tensor(self.__input_index, image)
        self.__interpreter.invoke()
        preds = self.__interpreter.get_tensor(self.__output_index)

        boxes = self.__postprocess(preds, orig_shape, pad_info, scale)
        return boxes, []
    
class YOLOSegment:
    def __init__(self):
        self.__interpreter = tf.lite.Interpreter(f"weights/yolo26n-seg_float32.tflite", num_threads=4)
        self.__interpreter.allocate_tensors()

        self.__input_index = self.__interpreter.get_input_details()[0]["index"]
        self.__output0_index = self.__interpreter.get_output_details()[0]["index"]
        self.__output1_index = self.__interpreter.get_output_details()[1]["index"]

        self.__target_size = 320
        self.classes = COCO_CLASSES

    # @line_profiler.profile
    def __preprocess(
        self,
        frame: npt.NDArray
    ) -> tuple[npt.NDArray, tuple[int, int], float, tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        scale = self.__target_size / max(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pad_h = self.__target_size - new_h
        pad_w = self.__target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = cv2.multiply(img, 1.0 / 255.0, dtype=cv2.CV_32F)
        img = np.expand_dims(img, axis=0)

        return img, frame.shape[:2], scale, (top, bottom, left, right)

    # @line_profiler.profile
    def __postprocess(
        self,
        preds: npt.NDArray, 
        proto: npt.NDArray, 
        orig_shape: tuple[int, int],
        scale: float, 
        pad_info: tuple[int, int, int, int]
    ) -> Objects:
        preds = preds[0]
        proto = proto[0]

        boxes = preds[:, :4]
        confs = preds[:, 4].astype(np.float32)
        cls_ids = preds[:, 5].astype(np.int32)
        coeffs = preds[:, 6:]

        filter_mask = confs >= 0.25
        boxes = boxes[filter_mask]
        confs = confs[filter_mask]
        cls_ids = cls_ids[filter_mask]
        coeffs = coeffs[filter_mask]

        h, w = orig_shape
        top, bottom, left, right = pad_info

        x1, y1, x2, y2 = boxes.T

        x1: npt.NDArray = np.clip((x1 * self.__target_size - left) / scale, 0, w).astype(np.int32)
        y1: npt.NDArray = np.clip((y1 * self.__target_size - top) / scale, 0, h).astype(np.int32)
        x2: npt.NDArray = np.clip((x2 * self.__target_size - left) / scale, 0, w).astype(np.int32)
        y2: npt.NDArray = np.clip((y2 * self.__target_size - top) / scale, 0, h).astype(np.int32)

        final_boxes = [
            (x1[i], y1[i], x2[i], y2[i], cls_ids[i], confs[i]) 
            for i in range(len(x1))
        ]

        proto_flat = proto.reshape(-1, 32)
        masks = proto_flat @ coeffs.T 
        masks: npt.NDArray = 1 / (1 + np.exp(-masks))
        masks = masks.reshape(80, 80, -1)
        masks = np.transpose(masks, (2, 0, 1))

        final_masks = []

        for i, mask in enumerate(masks):
            mask: npt.NDArray = cv2.resize(mask, (self.__target_size, self.__target_size), interpolation=cv2.INTER_LINEAR)
            mask = mask[top: self.__target_size - bottom, left: self.__target_size - right]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            cropped = np.zeros_like(mask, dtype=np.uint8)
            cropped[y1[i]: y2[i], x1[i]:x2[i]] = (mask[y1[i]: y2[i], x1[i]:x2[i]] > 0.5)
            final_masks.append(cropped.astype(bool))

        return final_boxes, final_masks
    
    # @line_profiler.profile
    def get_objects(
        self, 
        frame: npimagebgr
    ) -> Objects:
        image, orig_shape, scale, pad_info = self.__preprocess(frame)

        self.__interpreter.set_tensor(self.__input_index, image)
        self.__interpreter.invoke()
        preds = self.__interpreter.get_tensor(self.__output0_index)
        proto = self.__interpreter.get_tensor(self.__output1_index)

        boxes, masks = self.__postprocess(preds, proto, orig_shape, scale, pad_info)

        return boxes, masks

class ObjectDetectorTFLite:
    class ModelType(Enum):
        YOLO_DETECT  = 1
        YOLO_SEGMENT = 2

    def __init__(
        self, 
        overlay_ratio: float = 0.5,
        overlay_color: tuple[int, int, int] = (0, 255, 0),
        overlay_thickness: int = 2,
        overlay_font_scale: float = 0.5,
        overlay_centroid_radius: int = 10,
    ):
        self.models: dict[ObjectDetectorTFLite.ModelType, YOLODetect | YOLOSegment] = {
            self.ModelType.YOLO_DETECT: YOLODetect(),
            self.ModelType.YOLO_SEGMENT: YOLOSegment()
        }

        self.classes = {
            self.ModelType.YOLO_DETECT: self.models[self.ModelType.YOLO_DETECT].classes,
            self.ModelType.YOLO_SEGMENT: self.models[self.ModelType.YOLO_SEGMENT].classes
        }

        self.model_types = { model_type.value for model_type in ObjectDetectorTFLite.ModelType }

        self.model_type: None | ObjectDetectorTFLite.ModelType = None
        self.boxes: list[Box] = []
        self.masks: list[Mask] = []
        self.centroids: list[Centroid] = []

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
        frame: npimagebgr, 
        model_yolo_type: ObjectDetectorTFLite.ModelType
    ) -> tuple[list[Box], list[Mask], list[Centroid]]:
        self.model_type = model_yolo_type
        objects = self.models[self.model_type].get_objects(frame)

        self.boxes, self.masks = objects
        self.centroids = []
        
        if model_yolo_type == self.ModelType.YOLO_DETECT or len(self.masks) == 0:
            for box in self.boxes:
                x1, y1, x2, y2, cls, _ = box
                centroid = (int((x2 + x1) / 2), int((y2 + y1) / 2))
                self.centroids.append((centroid, self.classes[self.model_type][cls]))
        else:
            self.centroids = [
                (
                    tuple(map(round, ndimage.center_of_mass(mask)[::-1])), 
                    self.classes[self.model_type][box[4]]
                ) 
                for mask, box in zip(self.masks, self.boxes)
            ]
        
        return self.boxes, self.masks, self.centroids
    
    def get_object_depths(
        self,
        depth_bw: npt.NDArray
    ) -> list[int]:
        if self.model_type is None or len(self.boxes) == 0:
            return []

        depths: list[int] = []

        if self.model_type == self.ModelType.YOLO_DETECT or len(self.masks) == 0:
            for box in self.boxes:
                object_depth, _, _ = get_object_depth_box(depth_bw, box)
                depths.append(object_depth)
        else:
            for mask in self.masks:
                object_depth, _, _ = get_object_depth_mask(depth_bw, mask)
                depths.append(object_depth)
        
        return depths

    def get_warning_centroids(
        self,
        depth_bw: npt.NDArray,
        depth_warning_threshold: int = 150
    ) -> list[Centroid]:
        if self.model_type is None or len(self.boxes) == 0:
            return []

        warning_centroids: list[Centroid] = []

        if self.model_type == self.ModelType.YOLO_DETECT or len(self.masks) == 0:
            for box, centroid in zip(self.boxes, self.centroids):
                object_depth, _, _ = get_object_depth_box(depth_bw, box)
                if object_depth >= depth_warning_threshold:
                    warning_centroids.append(centroid)
        else:
            for mask, centroid in zip(self.masks, self.centroids):
                object_depth, _, _ = get_object_depth_mask(depth_bw, mask)
                if object_depth >= depth_warning_threshold:
                    warning_centroids.append(centroid)
        
        return warning_centroids
    
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
    ) -> list[Centroid]:
        if self.model_type is None or len(self.boxes) == 0:
            return []

        warning_centroids: list[Centroid] = []

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
