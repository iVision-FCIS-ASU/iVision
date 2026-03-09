from __future__ import annotations
import cv2
import torch
import numpy as np
import tensorflow as tf
from typing import Literal
from numpy import typing as npt
from .midas.model_loader import model_paths, load_model
from .midas.model_processor import process, create_side_by_side
from enum import Enum

class DepthEstimator:
    class ModelType(Enum):
        MIDAS_ORIG = 1
        MIDAS_V21 = 2
        DEPTH_ANYTHING_V2 = 3

    def __init__(self):
        # MIDAS_ORIG is kept temporarily until perfectly replicated in TFLITE
        self.models: dict[DepthEstimator.ModelType, MiDaS | MiDaSv21 | DepthAnythingV2] = {
            self.ModelType.MIDAS_ORIG: MiDaS("midas_v21_small_256"),
            self.ModelType.MIDAS_V21: MiDaSv21(),
            self.ModelType.DEPTH_ANYTHING_V2: DepthAnythingV2()
        }

        self.model_types = { model_type.value for model_type in DepthEstimator.ModelType }

    def get_depth_image(
        self, 
        frame: npt.NDArray, 
        model_type: DepthEstimator.ModelType
    ) -> tuple[npt.NDArray, npt.NDArray]:
        return self.models[model_type].get_depth_image(frame)

class MiDaS:
    def __init__(self, model_type: Literal["midas_v21_small_256", "dpt_swin2_tiny_256"]):
        self.model_type = model_type
        model_path = model_paths[self.model_type]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, model_path, model_type, False)
    
    def get_depth_image(self, frame):
        with torch.no_grad():
            original_image_rgb = np.flip(frame, 2)
            image = self.transform({"image": original_image_rgb/255})["image"]

            prediction = process(self.device, self.model, self.model_type, 
                                image, (self.net_w, self.net_h),
                                original_image_rgb.shape[1::-1], False, True)
            
            return create_side_by_side(None, prediction, False)

class MiDaSv21:
    def __init__(self):
        self.target_size = 256
        self.interpreter = tf.lite.Interpreter(f"weights/midas_v21_small_256.tflite", num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
    
    def __preprocess(
        self,
        frame: npt.NDArray
    ) -> tuple[npt.NDArray, tuple[int, int], tuple[int, int, int, int]]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        h, w = frame.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        # new_h, new_w = h * scale, w * scale
        # multiple = 32
        # new_h = int(round(new_h / multiple) * multiple)
        # new_w = int(round(new_w / multiple) * multiple)
        # image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = np.expand_dims(image, axis=0)

        return image, frame.shape[:2], (top, bottom, left, right)
    
    def __postprocess(
        self,
        depth_raw: npt.NDArray, 
        orig_shape: tuple[int, int], 
        pad_info: tuple[int, int, int, int]
    ) -> npt.NDArray:
        depth_img = np.squeeze(depth_raw)
        
        top, bottom, left, right = pad_info
        depth_img = depth_img[top:self.target_size - bottom, left:self.target_size - right]

        depth_img = cv2.resize(depth_img, orig_shape[::-1], interpolation=cv2.INTER_CUBIC)
        depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return depth_img
    
    def get_depth_image(self, frame: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        image, orig_shape, pad_info = self.__preprocess(frame)

        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        depth_raw = self.interpreter.get_tensor(self.output_index)

        depth_bw = self.__postprocess(depth_raw, orig_shape, pad_info)
        depth_bgr = cv2.applyColorMap(depth_bw, cv2.COLORMAP_JET)

        return depth_bw, depth_bgr

class DepthAnythingV2:
    def __init__(self):
        self.target_size = 224 # 224/252/378/518
        self.interpreter = tf.lite.Interpreter(f"weights/depth_anything_v2_{self.target_size}.tflite", num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def __preprocess(
        self,
        frame: npt.NDArray
    ) -> tuple[npt.NDArray, tuple[int, int], tuple[int, int, int, int]]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        h, w = frame.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)

        image = np.expand_dims(image, axis=0)

        return image, frame.shape[:2], (top, bottom, left, right)

    def __postprocess(
        self,
        depth_raw: npt.NDArray, 
        orig_shape: tuple[int, int], 
        pad_info: tuple[int, int, int, int]
    ) -> npt.NDArray:
        depth_img = np.squeeze(depth_raw)

        top, bottom, left, right = pad_info
        depth_img = depth_img[top:self.target_size - bottom, left:self.target_size - right]
        
        depth_img = cv2.resize(depth_img, orig_shape[::-1], interpolation=cv2.INTER_CUBIC)
        depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return depth_img

    def get_depth_image(self, frame: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        image, orig_shape, pad_info = self.__preprocess(frame)

        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        depth_raw = self.interpreter.get_tensor(self.output_index)

        depth_bw = self.__postprocess(depth_raw, orig_shape, pad_info)
        depth_bgr = cv2.applyColorMap(depth_bw, cv2.COLORMAP_JET)

        return depth_bw, depth_bgr
