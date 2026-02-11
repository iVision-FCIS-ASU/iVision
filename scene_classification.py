import cv2
import timm
import torch
import line_profiler
from PIL import Image
from numpy import typing as npt
import torchvision.transforms as transforms
from utils.scene_mapping import INDOOR_CLASSES, OUTDOOR_MERGE_RULES

class SceneClassifier:
    def __init__(self):
        print("\n-----SCENE CLASSIFIER INITIALIZATION-----")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_binary = self.__load_mobilevit_weights('weights/best_mobilevit_merged.pth', num_classes=2)
        self.model_indoor = self.__load_mobilevit_weights('weights/best_mobilevit_indoor.pth', num_classes=len(INDOOR_CLASSES))
        self.model_outdoor = self.__load_mobilevit_weights('weights/best_mobilevit_outdoor.pth', num_classes=len(OUTDOOR_MERGE_RULES))

        self.models = { 0: self.model_indoor, 1: self.model_outdoor }
        self.scenes_binary = { 0: "indoor", 1: "outdoor" }
        self.scenes = { 0: list(INDOOR_CLASSES.keys()), 1: list(OUTDOOR_MERGE_RULES.keys()) }
        
        print("-----SCENE CLASSIFIER INITIALIZED-----\n")

    def __load_mobilevit_weights(self, model_path: str, num_classes: int) -> torch.nn.Module:
        print(f"Loading weights for: {model_path}")
        model: torch.nn.Module = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def __preprocess(self, frame: npt.NDArray) -> torch.Tensor:
        raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_img = Image.fromarray(raw_img)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_img = transform(raw_img).unsqueeze(0).to(self.device)
        return tensor_img
    
    @torch.inference_mode()
    def get_scene_type_binary(self, frame: npt.NDArray) -> str:
        tensor_img = self.__preprocess(frame)
        is_outdoor = torch.argmax(self.model_binary(tensor_img), dim=1).item()
        return self.scenes_binary[is_outdoor]
    
    @torch.inference_mode()
    def get_scene_type(self, frame: npt.NDArray) -> tuple[str, str]:
        tensor_img = self.__preprocess(frame)
        is_outdoor = torch.argmax(self.model_binary(tensor_img), dim=1).item()
        scene_idx = torch.argmax(self.models[is_outdoor](tensor_img), dim=1).item()
        scene_label = self.scenes[is_outdoor][scene_idx]
        return self.scenes_binary[is_outdoor], scene_label
