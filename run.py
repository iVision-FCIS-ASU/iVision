import cv2
import time
import torch
import numpy as np
import threading
import line_profiler
from typing import Literal
from numpy import typing as npt
from datetime import datetime
from models_download import download_models
from modules.object_detection import ObjectDetector
from modules.depth_estimation import DepthEstimator
from modules.complexity_estimation import SceneType, Weather, Complexity, ComplexityEstimator
from modules.scene_classification import SceneClassifier
from modules.scene_narration import SceneNarrator

class iVision:
    def __init__(
        self, 
        model_yolo_type: tuple[ObjectDetector.ModelType, ObjectDetector.ModelType],
        model_depth_type: tuple[DepthEstimator.ModelType, DepthEstimator.ModelType],
        side_by_side: bool = False,
        debug: bool = False,
        cap_id: int = 0
    ):
        print("\n=========================")
        print("-----iVision Started-----")
        print("=========================\n")

        self.model_yolo_type_simple, self.model_yolo_type_complex = model_yolo_type
        self.model_depth_type_simple, self.model_depth_type_complex = model_depth_type
        self.side_by_side = side_by_side
        self.debug = debug
        self.cap_id = cap_id
        
        download_models()
        self.__get_models()
        
        self.complexity_lock = threading.Lock()
        self.complexity = Complexity.SIMPLE
        self.scene_type = SceneType.INDOOR
        self.weather = Weather.BRIGHT

        self.frame_lock = threading.Lock()
        self.frame = np.zeros(1)
        self.IS_RUNNING = True
        self.IS_CAPTION_RUNNING = False
        self.__run()
        
        print("\n========================")
        print("-----iVision Closed-----")
        print("========================\n")

    def __get_models(self):
        print("\n-----Loading Models-----\n")

        print("-----Loading YOLO-----")
        self.model_object_detector = ObjectDetector()
        self.models_yolo = {
            Complexity.SIMPLE: self.model_yolo_type_simple,
            Complexity.COMPLEX: self.model_yolo_type_complex
        }
        
        print("-----Loading Depth Estimation-----")
        self.model_depth_estimator = DepthEstimator()
        self.models_depth = {
            Complexity.SIMPLE: self.model_depth_type_simple,
            Complexity.COMPLEX: self.model_depth_type_complex
        }
        
        print("-----Loading Complexity Estimation-----")
        self.model_complexity_estimator = ComplexityEstimator()
        
        print("-----Loading Scene Classifier-----")
        self.model_scene_classifier = SceneClassifier()

        print("-----Loading Scene Narration-----")
        self.model_scene_narrator = SceneNarrator()
        
        print("\n-----All Models Loaded-----\n")

    def __get_output_image(self, frame: npt.NDArray):
        with self.complexity_lock:
            complexity = self.complexity
            scene_type = self.scene_type
            weather = self.weather
        
        depth_bw, depth_rgb = self.model_depth_estimator.get_depth_image(frame, self.models_depth[complexity])
        boxes, masks, centroids = self.model_object_detector.get_objects(frame, self.models_yolo[complexity])
        
        # output_image = depth_rgb.copy()
        output_image = cv2.cvtColor(depth_bw, cv2.COLOR_GRAY2BGR)
        self.model_object_detector.draw_objects_with_depth(output_image, depth_bw, True)
        
        if not self.side_by_side:
            return output_image

        yolo_image = frame.copy()
        cv2.putText(frame, f"{complexity.name}, {scene_type.name}, {weather.name}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.model_object_detector.draw_objects(yolo_image)
        output_image = np.vstack((np.hstack((frame, yolo_image)), 
                                  np.hstack((cv2.cvtColor(depth_bw, cv2.COLOR_GRAY2BGR), output_image))))

        return output_image

    def __camera_thread(self):
        print("-----Starting Camera Thread-----")
        cap = cv2.VideoCapture(self.cap_id, cv2.CAP_DSHOW)

        while self.IS_RUNNING:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to capture frame!")
                self.IS_RUNNING = False
                break
            
            with self.frame_lock:
                self.frame = frame
            # time.sleep(0.001)
            time.sleep(0)
        
        cap.release()
        print("-----Stopped Camera Thread-----")

    def __complexity_estimation_thread(self):
        print("-----Starting Complexity Estimation Thread-----")
        SLEEP_COUNT = 10
        SLEEP_SECS = 5 / SLEEP_COUNT
        MAX_PREDICTIONS = 5
        predictions_counter = 0
        
        scene_type_map = {"indoor": SceneType.INDOOR, "outdoor": SceneType.OUTDOOR}
        scene_types_dict: dict[SceneType, int] = {}
        weathers_dict: dict[Weather, int] = {}
        complexities_dict: dict[Complexity, int] = {}
        
        prev_frame = np.zeros(1)

        while self.IS_RUNNING:
            with self.frame_lock:
                frame = self.frame.copy()
            
            # if frame is None or np.array_equal(frame, prev_frame):
            if np.array_equal(frame, prev_frame):
                # time.sleep(0.001)
                time.sleep(0)
                continue
            prev_frame = frame

            scene_type = scene_type_map[self.model_scene_classifier.get_scene_type_binary(frame)]
            complexity, weather, confidence = self.model_complexity_estimator.predict(scene_type, frame)

            # if complexity not in predictions_dict:
            #     predictions_dict[complexity] = 0
            # predictions_dict[complexity] += 1

            # if scene_type not in scene_types_dict:
            #     scene_types_dict[scene_type] = 0
            # scene_types_dict[scene_type] += 1

            if weather not in weathers_dict:
                weathers_dict[weather] = 0
            weathers_dict[weather] += 1

            predictions_counter += 1
            if predictions_counter >= MAX_PREDICTIONS:
                predictions_counter = 0
                # max_complexity: Complexity = max(predictions_dict, key=predictions_dict.get)
                # max_scene_type: SceneType = max(scene_types_dict, key=scene_types_dict.get)
                max_weather: Weather = max(weathers_dict, key=weathers_dict.get)
                max_scene_type = self.model_complexity_estimator.weather_to_scene[max_weather]
                max_complexity = self.model_complexity_estimator.weather_to_complexity[max_weather]
                # complexities_dict.clear()
                # scene_types_dict.clear()
                weathers_dict.clear()


                cur_datetime = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                # print(f"\nScene: {max_scene_type}\nWeather: {max_weather.name}\nComplexity: {max_complexity.name}\n")
                print((f"\n{cur_datetime} | Complexity: {max_complexity.name}\n"
                       f"{cur_datetime} | Scene: {max_scene_type.name}\n"
                       f"{cur_datetime} | Weather: {max_weather.name}\n"))
                
                with self.complexity_lock:
                    self.complexity = max_complexity
                    self.scene_type = max_scene_type
                    self.weather = max_weather
                
                # time.sleep(SLEEP_SECS)
                for _ in range(SLEEP_COUNT):
                    if not self.IS_RUNNING:
                        break
                    time.sleep(SLEEP_SECS)
        
        print("-----Stopped Complexity Estimation Thread-----")

    def __captioning_thread(self, frame):
        print("\n-----STARTING NARRATION-----")
        narration_text = self.model_scene_narrator.get_narration(frame, self.model_scene_classifier)
        print(f"Final Narration: {narration_text}")
        print("-----STOPPING NARRATION-----")
        self.IS_CAPTION_RUNNING = False

    def __run(self):
        print("\n-----Starting Main Thread-----\n")
        camera_thread = threading.Thread(target=self.__camera_thread, daemon=True)
        camera_thread.start()
        complexity_estimation_thread = threading.Thread(target=self.__complexity_estimation_thread, daemon=True)
        complexity_estimation_thread.start()
        prev_frame = np.zeros(1)

        print("-----Main Loop Started-----")
        while self.IS_RUNNING:
            with self.frame_lock:
                frame = self.frame.copy()
            
            # if frame is None or np.array_equal(frame, prev_frame):
            if np.array_equal(frame, prev_frame):
                # time.sleep(0.001)
                time.sleep(0)
                continue
            prev_frame = frame
            
            output_image = self.__get_output_image(frame)
            cv2.imshow(f"Object Detection + Depth Estimation", output_image)

            key_pressed = cv2.waitKey(1)

            if not self.IS_CAPTION_RUNNING and (key_pressed == ord("c") or key_pressed == ord("C")):
                self.IS_CAPTION_RUNNING = True
                threading.Thread(target=self.__captioning_thread, args=(frame,), daemon=True).start()
            
            if key_pressed == ord("q") or key_pressed == ord("Q"):
                break

        print("-----Stopping Main Thread-----")
        self.IS_RUNNING = False
        camera_thread.join()
        complexity_estimation_thread.join()
        cv2.destroyAllWindows()
        print("\n-----Stopped Main Thread-----\n")

if __name__ == "__main__":
    iVision(
        model_yolo_type=(ObjectDetector.ModelType.YOLO_SEGMENT, ObjectDetector.ModelType.YOLO_SEGMENT),
        model_depth_type=(DepthEstimator.ModelType.DEPTH_ANYTHING_V2, DepthEstimator.ModelType.DEPTH_ANYTHING_V2),
        side_by_side=True,
        debug=False,
        cap_id=0
    )
