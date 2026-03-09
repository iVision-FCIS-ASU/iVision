import cv2
import timm
import torch
import numpy as np
import line_profiler
import tensorflow as tf
import torchvision.transforms as transforms
from PIL import Image
from numpy import typing as npt
from .utils.scene_mapping import INDOOR_CLASSES, OUTDOOR_MERGE_RULES

class SceneClassifier:
    def __init__(self):
        print("\n-----SCENE CLASSIFIER INITIALIZATION-----")
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model_binary = self.__load_mobilevit_weights('weights/best_mobilevit_merged.pth', num_classes=2)
        # self.model_indoor = self.__load_mobilevit_weights('weights/best_mobilevit_indoor.pth', num_classes=len(INDOOR_CLASSES))
        # self.model_outdoor = self.__load_mobilevit_weights('weights/best_mobilevit_outdoor.pth', num_classes=len(OUTDOOR_MERGE_RULES))

        # self.models = { 0: self.model_indoor, 1: self.model_outdoor }
        self.scenes_binary = { 0: "indoor", 1: "outdoor" }
        self.scenes = { 0: list(INDOOR_CLASSES.keys()), 1: list(OUTDOOR_MERGE_RULES.keys()) }

        self.interpreter_binary = tf.lite.Interpreter(f"weights/best_mobilevit_merged.tflite", num_threads=4)
        self.interpreter_binary.allocate_tensors()
        self.input_index_binary = self.interpreter_binary.get_input_details()[0]["index"]
        self.output_index_binary = self.interpreter_binary.get_output_details()[0]["index"]

        self.interpreter_indoor = tf.lite.Interpreter(f"weights/best_mobilevit_indoor.tflite", num_threads=4)
        self.interpreter_indoor.allocate_tensors()
        self.input_index_indoor = self.interpreter_indoor.get_input_details()[0]["index"]
        self.output_index_indoor = self.interpreter_indoor.get_output_details()[0]["index"]

        self.interpreter_outdoor = tf.lite.Interpreter(f"weights/best_mobilevit_outdoor.tflite", num_threads=4)
        self.interpreter_outdoor.allocate_tensors()
        self.input_index_outdoor = self.interpreter_outdoor.get_input_details()[0]["index"]
        self.output_index_outdoor = self.interpreter_outdoor.get_output_details()[0]["index"]

        self.models = {
            0: (self.interpreter_indoor, self.input_index_indoor, self.output_index_indoor),
            1: (self.interpreter_outdoor, self.input_index_outdoor, self.output_index_outdoor)
        }
        
        print("-----SCENE CLASSIFIER INITIALIZED-----\n")

    # def __load_mobilevit_weights(self, model_path: str, num_classes: int) -> torch.nn.Module:
    #     print(f"Loading weights for: {model_path}")
    #     model: torch.nn.Module = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=num_classes)
    #     state_dict = torch.load(model_path, map_location=self.device)
    #     model.load_state_dict(state_dict)
    #     model.to(self.device)
    #     model.eval()
    #     return model
    
    # @line_profiler.profile
    # def __preprocess(self, frame: npt.NDArray) -> torch.Tensor:
    #     raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     raw_img = Image.fromarray(raw_img)
    #     transform = transforms.Compose([
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     tensor_img = transform(raw_img).unsqueeze(0).to(self.device)
    #     return tensor_img
    
    # @line_profiler.profile
    def __preprocess(self, frame: npt.NDArray) -> npt.NDArray:
        raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        raw_img = Image.fromarray(raw_img)
        raw_img = raw_img.resize((256,256), resample=Image.BILINEAR)
        raw_img = np.array(raw_img, dtype=np.float32) / 255.0

        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        raw_img -= mean
        raw_img /= std

        raw_img = np.expand_dims(raw_img, axis=0)
        
        return raw_img
    
    # @torch.inference_mode()
    # @line_profiler.profile
    # def get_scene_type_binary(self, frame: npt.NDArray) -> str:
    #     tensor_img = self.__preprocess(frame)
    #     is_outdoor = torch.argmax(self.model_binary(tensor_img), dim=1).item()
    #     return self.scenes_binary[is_outdoor]
    
    # @line_profiler.profile
    def get_scene_type_binary(self, frame: npt.NDArray) -> str:
        input_img = self.__preprocess(frame)
        
        self.interpreter_binary.set_tensor(self.input_index_binary, input_img)
        self.interpreter_binary.invoke()
        preds = self.interpreter_binary.get_tensor(self.output_index_binary)
        
        is_outdoor = np.argmax(preds, axis=1)[0]
        return self.scenes_binary[is_outdoor]
    
    # @torch.inference_mode()
    # @line_profiler.profile
    # def get_scene_type(self, frame: npt.NDArray) -> tuple[str, str]:
    #     tensor_img = self.__preprocess(frame)
    #     is_outdoor = torch.argmax(self.model_binary(tensor_img), dim=1).item()
    #     scene_idx = torch.argmax(self.models[is_outdoor](tensor_img), dim=1).item()
    #     scene_label = self.scenes[is_outdoor][scene_idx]
    #     return self.scenes_binary[is_outdoor], scene_label
    
    # @line_profiler.profile
    def get_scene_type(self, frame: npt.NDArray) -> tuple[str, str]:
        input_img = self.__preprocess(frame)
        
        self.interpreter_binary.set_tensor(self.input_index_binary, input_img)
        self.interpreter_binary.invoke()
        preds_binary = self.interpreter_binary.get_tensor(self.output_index_binary)
        is_outdoor = np.argmax(preds_binary, axis=1)[0]
        
        interpreter, input_index, output_index = self.models[is_outdoor]
        interpreter.set_tensor(input_index, input_img)
        interpreter.invoke()
        preds_scene = interpreter.get_tensor(output_index)

        scene_index = np.argmax(preds_scene, axis=1)[0]
        scene_label = self.scenes[is_outdoor][scene_index]
        return self.scenes_binary[is_outdoor], scene_label
