import cv2
import numpy as np
from enum import Enum, auto
from tensorflow.keras.models import load_model

class SceneType(Enum):
    INDOOR = auto()
    OUTDOOR = auto()

class Weather(Enum):
    CLOUDY = auto() 
    FOGGY = auto()
    NIGHT = auto()
    RAINY = auto()
    SNOWY = auto()
    SUNNY = auto()
    UNKNOWN = auto()

class Complexity(Enum):
    SIMPLE = auto()
    COMPLEX = auto()

class ComplexityEstimation:
    def __init__(self):
        self.model = load_model("weights/complexity_estimation_v1.keras")
        self.classes = [Weather.CLOUDY, Weather.FOGGY, Weather.NIGHT, Weather.RAINY, Weather.SNOWY, Weather.SUNNY]
        self.complexity_mapping = {
            Weather.CLOUDY: Complexity.SIMPLE,
            Weather.FOGGY:  Complexity.COMPLEX,
            Weather.NIGHT:  Complexity.COMPLEX,
            Weather.RAINY:  Complexity.COMPLEX,
            Weather.SNOWY:  Complexity.SIMPLE,
            Weather.SUNNY:  Complexity.SIMPLE,
            Weather.UNKNOWN: Complexity.SIMPLE
        }
    
    def __preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype(np.uint8)
        frame = np.expand_dims(frame, axis=0)
        return frame

    def __predict_indoor(self, frame):
        "NEEDS UPDATING WITH PROPER METHOD"
        return Complexity.SIMPLE, Weather.UNKNOWN, 1.0

    def __predict_outdoor(self, frame):
        input_img = self.__preprocess(frame)
    
        preds = self.model.predict(input_img, verbose=0)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = preds[0][class_id]
        label = self.classes[class_id]
        complexity = self.complexity_mapping[label]

        return complexity, label, confidence
    
    def predict(self, scene_type_binary: SceneType, frame):
        if scene_type_binary == SceneType.OUTDOOR:
            return self.__predict_outdoor(frame)
        else:
            return self.__predict_indoor(frame)
    
    