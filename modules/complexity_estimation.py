import cv2
import numpy as np
import tensorflow as tf
import line_profiler
from enum import Enum, auto
from numpy import typing as npt

class SceneType(Enum):
    INDOOR  = auto()
    OUTDOOR = auto()

class Weather(Enum):
    CLOUDY  = auto() 
    FOGGY   = auto()
    NIGHT   = auto()
    RAINY   = auto()
    SNOWY   = auto()
    SUNNY   = auto()
    UNKNOWN = auto()

class Complexity(Enum):
    SIMPLE  = auto()
    COMPLEX = auto()

class ComplexityEstimator:
    def __init__(self):
        print("\n-----COMPLEXITY ESTIMATION INITIALIZATION-----")
        
        self.interpreter = tf.lite.Interpreter("weights/complexity_estimation_v1.tflite")
        self.interpreter.allocate_tensors()
        self.input_details_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details_index = self.interpreter.get_output_details()[0]["index"]

        self.classes = [Weather.CLOUDY, Weather.FOGGY, Weather.NIGHT, Weather.RAINY, Weather.SNOWY, Weather.SUNNY]
        self.complexity_mapping = {
            Weather.CLOUDY:  Complexity.SIMPLE,
            Weather.FOGGY:   Complexity.COMPLEX,
            Weather.NIGHT:   Complexity.COMPLEX,
            Weather.RAINY:   Complexity.COMPLEX,
            Weather.SNOWY:   Complexity.SIMPLE,
            Weather.SUNNY:   Complexity.SIMPLE,
            Weather.UNKNOWN: Complexity.SIMPLE
        }
    
        print("\n-----COMPLEXITY ESTIMATION INITIALIZED-----")

    def __preprocess(self, frame: npt.NDArray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype(np.float32)
        frame = np.expand_dims(frame, axis=0)
        return frame

    def __predict_indoor(self, frame: npt.NDArray):
        "NEEDS UPDATING WITH PROPER METHOD"
        return Complexity.SIMPLE, Weather.UNKNOWN, 1.0

    @line_profiler.profile
    def __predict_outdoor(self, frame: npt.NDArray):
        input_img = self.__preprocess(frame)
    
        self.interpreter.set_tensor(self.input_details_index, input_img)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details_index)

        class_id = np.argmax(preds, axis=1)[0]
        confidence = preds[0][class_id]
        weather = self.classes[class_id]
        complexity = self.complexity_mapping[weather]

        return complexity, weather, confidence
    
    def predict(self, scene_type_binary: SceneType, frame: npt.NDArray):
        if scene_type_binary == SceneType.OUTDOOR:
            return self.__predict_outdoor(frame)
        else:
            return self.__predict_indoor(frame)
