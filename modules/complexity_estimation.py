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
    BRIGHT  = auto()
    DARK    = auto()

class Complexity(Enum):
    SIMPLE  = auto()
    COMPLEX = auto()

class ComplexityEstimator:
    def __init__(self):
        print("\n-----COMPLEXITY ESTIMATION INITIALIZATION-----")
        
        self.interpreter = tf.lite.Interpreter("weights/complexity_estimation_v1.tflite", num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details_index = self.interpreter.get_output_details()[0]["index"]

        self.classes = [Weather.CLOUDY, Weather.FOGGY, Weather.NIGHT, Weather.RAINY, Weather.SNOWY, Weather.SUNNY]
        self.weather_to_complexity = {
            Weather.CLOUDY:  Complexity.SIMPLE,
            Weather.FOGGY:   Complexity.COMPLEX,
            Weather.NIGHT:   Complexity.COMPLEX,
            Weather.RAINY:   Complexity.COMPLEX,
            Weather.SNOWY:   Complexity.SIMPLE,
            Weather.SUNNY:   Complexity.SIMPLE,
            Weather.BRIGHT:  Complexity.SIMPLE,
            Weather.DARK:    Complexity.COMPLEX
        }
        self.weather_to_scene = {
            Weather.CLOUDY:  SceneType.OUTDOOR,
            Weather.FOGGY:   SceneType.OUTDOOR,
            Weather.NIGHT:   SceneType.OUTDOOR,
            Weather.RAINY:   SceneType.OUTDOOR,
            Weather.SNOWY:   SceneType.OUTDOOR,
            Weather.SUNNY:   SceneType.OUTDOOR,
            Weather.BRIGHT:  SceneType.INDOOR,
            Weather.DARK:    SceneType.INDOOR
        }
    
        print("\n-----COMPLEXITY ESTIMATION INITIALIZED-----")

    # @line_profiler.profile
    def __preprocess(self, frame: npt.NDArray):
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (224, 224)).astype(np.float32)
        input_img = np.expand_dims(input_img, axis=0)
        return input_img

    # @line_profiler.profile
    def __predict_indoor(self, frame: npt.NDArray) -> tuple[Complexity, Weather, float]:
        # frame_lum = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
        lum_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        complexity_threshold = 150
        percentile_threshold = 85
        lum = int(np.percentile(lum_img, percentile_threshold))
        # # maybe consider contrast too (although I doubt it will help)
        # contrast = int(np.std(frame_lum))
        
        if complexity_threshold == 255:
            simple_confidence = float(lum == 255)
        else:
            simple_confidence = 0.5 * ((lum - complexity_threshold) / (255 - complexity_threshold)) + 0.5

        if complexity_threshold == 0:
            complex_confidence = float(lum == 0)
        else:
            complex_confidence = 0.5 * ((complexity_threshold - lum) / complexity_threshold) + 0.5
        
        if lum >= complexity_threshold:
            return Complexity.SIMPLE, Weather.BRIGHT, simple_confidence
        else:
            return Complexity.COMPLEX, Weather.DARK, complex_confidence

    # @line_profiler.profile
    def __predict_outdoor(self, frame: npt.NDArray) -> tuple[Complexity, Weather, float]:
        input_img = self.__preprocess(frame)
    
        self.interpreter.set_tensor(self.input_details_index, input_img)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details_index)

        class_id = np.argmax(preds, axis=1)[0]
        confidence = preds[0][class_id]
        weather = self.classes[class_id]
        complexity = self.weather_to_complexity[weather]

        return complexity, weather, confidence
    
    def predict(self, scene_type_binary: SceneType, frame: npt.NDArray) -> tuple[Complexity, Weather, float]:
        if scene_type_binary == SceneType.OUTDOOR:
            return self.__predict_outdoor(frame)
        else:
            return self.__predict_indoor(frame)
