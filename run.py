import cv2
import time
import torch
import numpy as np
import threading
import line_profiler
from typing import Literal
from numpy import typing as npt
from models_download import download_models
from modules.object_detection import ObjectDetector
from modules.depth_estimation import MiDaS, DepthAnythingV2
from modules.complexity_estimation import SceneType, Weather, Complexity, ComplexityEstimator
from modules.scene_classification import SceneClassifier
from modules.scene_narration import SceneNarrator

class iVision:
    def __init__(
        self, 
        model_yolo_type: Literal["detect", "segment"],
        model_depth_type: Literal["midas_v21_small_256", "dpt_swin2_tiny_256", "depth_anything_v2"],
        side_by_side: bool = False,
        debug: bool = False,
        cap_id: int = 0
    ):
        print("\n=========================")
        print("-----iVision Started-----")
        print("=========================\n")

        self.model_yolo_type = model_yolo_type
        self.model_depth_type = model_depth_type
        self.side_by_side = side_by_side
        self.debug = debug
        self.cap_id = cap_id
        
        download_models()
        self.__get_models()
        
        self.complexity_lock = threading.Lock()
        self.complexity = Complexity.SIMPLE
        self.scene_type = SceneType.INDOOR
        self.weather = Weather.UNKNOWN

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
        yolo_types = ["detect", "segment"]
        if self.model_yolo_type not in yolo_types:
            print("ERROR: Invalid YOLO Model!")
            assert False
        
        depth_types = ["midas_v21_small_256", "dpt_swin2_tiny_256", "depth_anything_v2"]
        if self.model_depth_type not in depth_types:
            print("ERROR: Invalid Depth Model!")
            assert False

        print("-----Loading YOLO-----")
        self.model_object_detector = ObjectDetector()

        self.models_yolo = {
            Complexity.SIMPLE: ObjectDetector.ModelType.YOLO_SEGMENT,
            Complexity.COMPLEX: ObjectDetector.ModelType.YOLO_SEGMENT
        }
        
        print("-----Loading Depth Estimation-----")
        self.models_depth = {
            # Complexity.SIMPLE: MiDaS("midas_v21_small_256"),
            Complexity.SIMPLE: DepthAnythingV2(),
            # Complexity.COMPLEX: MiDaS("dpt_swin2_tiny_256")
        }

        match self.model_depth_type:
            case "midas_v21_small_256":
                self.models_depth[Complexity.COMPLEX] = MiDaS(self.model_depth_type)
            case "dpt_swin2_tiny_256":
                self.models_depth[Complexity.COMPLEX] = MiDaS(self.model_depth_type)
            case "depth_anything_v2":
                self.models_depth[Complexity.COMPLEX] = DepthAnythingV2()
        
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
        
        depth_bw, depth_rgb = self.models_depth[complexity].get_depth_image(frame)
        boxes, masks, centroids = self.model_object_detector.get_objects(frame, self.models_yolo[complexity])
        
        yolo_image = frame.copy()
        # output_image = depth_rgb.copy()
        output_image = cv2.cvtColor(depth_bw, cv2.COLOR_GRAY2BGR)

        cv2.putText(frame, f"{complexity.name}, {scene_type.name}, {weather.name}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.model_object_detector.draw_objects(yolo_image)
        # self.model_object_detector.draw_objects_with_depth(output_image, depth_bw)
        self.model_object_detector.draw_objects_with_depth(output_image, depth_bw, True)

        if self.side_by_side:
            # output_image = np.vstack((np.hstack((frame, yolo_image)), np.hstack((depth_rgb, output_image))))
            output_image = np.vstack((np.hstack((frame, yolo_image)), np.hstack((cv2.cvtColor(depth_bw, cv2.COLOR_GRAY2BGR), output_image))))

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
        SLEEP_SECS = 10 / SLEEP_COUNT
        MAX_PREDICTIONS = 5
        predictions_counter = 0
        
        scene_type_map = {"indoor": SceneType.INDOOR, "outdoor": SceneType.OUTDOOR}
        scene_types_dict: dict[SceneType, int] = {}
        weathers_dict: dict[str, int] = {}
        predictions_dict: dict[Complexity, int] = {}
        
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

            if scene_type not in scene_types_dict:
                scene_types_dict[scene_type] = 0
            scene_types_dict[scene_type] += 1

            if weather not in weathers_dict:
                weathers_dict[weather] = 0
            weathers_dict[weather] += 1

            if complexity not in predictions_dict:
                predictions_dict[complexity] = 0
            predictions_dict[complexity] += 1

            predictions_counter += 1
            if predictions_counter >= MAX_PREDICTIONS:
                predictions_counter = 0
                max_scene_type: str = max(scene_types_dict, key=scene_types_dict.get)
                max_weather: Weather = max(weathers_dict, key=weathers_dict.get)
                max_complexity: Complexity = max(predictions_dict, key=predictions_dict.get)
                predictions_dict.clear()

                print(f"\nScene: {max_scene_type}\nWeather: {max_weather.name}\nComplexity: {max_complexity.name}\n")
                
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
        model_yolo_type="segment",
        model_depth_type="depth_anything_v2",
        side_by_side=True,
        debug=False,
        cap_id=0
    )
