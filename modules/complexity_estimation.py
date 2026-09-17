import cv2
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from enum import auto, Enum

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
        
        self.__interpreter = tf.lite.Interpreter("weights/complexity_estimation_v1.tflite", num_threads=4)
        self.__interpreter.allocate_tensors()
        self.__input_details_index = self.__interpreter.get_input_details()[0]["index"]
        self.__output_details_index = self.__interpreter.get_output_details()[0]["index"]

        self.__classes = [Weather.CLOUDY, Weather.FOGGY, Weather.NIGHT, Weather.RAINY, Weather.SNOWY, Weather.SUNNY]
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

    def __preprocess(self, frame: npt.NDArray):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img
    
    def __postprocess(self, preds: npt.NDArray) -> tuple[Complexity, Weather, float]:
        class_id = np.argmax(preds, axis=1)[0]
        conf = preds[0][class_id]
        weather = self.__classes[class_id]
        complexity = self.weather_to_complexity[weather]
        return complexity, weather, conf

    def __predict_outdoor(self, frame: npt.NDArray) -> tuple[Complexity, Weather, float]:
        img = self.__preprocess(frame)
    
        self.__interpreter.set_tensor(self.__input_details_index, img)
        self.__interpreter.invoke()
        preds = self.__interpreter.get_tensor(self.__output_details_index)

        return self.__postprocess(preds)

    def __predict_indoor(self, frame: npt.NDArray) -> tuple[Complexity, Weather, float]:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lum_thresh = 150
        percentile_thresh = 85
        lum = int(np.percentile(img_gray, percentile_thresh))
        
        if lum_thresh == 255:
            simple_conf = float(lum == 255)
        else:
            simple_conf = 0.5 * ((lum - lum_thresh) / (255 - lum_thresh)) + 0.5

        if lum_thresh == 0:
            complex_conf = float(lum == 0)
        else:
            complex_conf = 0.5 * ((lum_thresh - lum) / lum_thresh) + 0.5
        
        if lum >= lum_thresh:
            return Complexity.SIMPLE, Weather.BRIGHT, simple_conf
        else:
            return Complexity.COMPLEX, Weather.DARK, complex_conf
    
    def predict(self, frame: npt.NDArray, scene_type_binary: SceneType) -> tuple[Complexity, Weather, float]:
        if scene_type_binary == SceneType.OUTDOOR:
            return self.__predict_outdoor(frame)
        else:
            return self.__predict_indoor(frame)
