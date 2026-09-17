import cv2
import keyboard
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
from modules.object_detection_tflite import ObjectDetectorTFLite
from modules.depth_estimation import DepthEstimator
from modules.complexity_estimation import SceneType, Weather, Complexity, ComplexityEstimator
from modules.scene_classification import SceneClassifier
from modules.scene_narration import SceneNarrator
from modules.utils.sliding_window import SlidingWindow
from modules.grid_obstacle_detection import GridObstacleDetector
from modules.grid_warning_system import GridWarningSystem
from modules.reading_mode import ReadingMode

class iVision:
    def __init__(
        self, 
        model_yolo_type: tuple[ObjectDetectorTFLite.ModelType, ObjectDetectorTFLite.ModelType],
        model_depth_type: tuple[DepthEstimator.ModelType, DepthEstimator.ModelType],
        depth_warning_threshold: int = 180,
        side_by_side: bool = False,
        debug: bool = False,
        cap_id: int = 0
    ):
        print("\n=========================")
        print("-----iVision Started-----")
        print("=========================\n")

        self.model_yolo_type_simple, self.model_yolo_type_complex = model_yolo_type
        self.model_depth_type_simple, self.model_depth_type_complex = model_depth_type
        self.depth_warning_threshold = depth_warning_threshold
        self.side_by_side = side_by_side
        self.debug = debug
        self.cap_id = cap_id

        cap = cv2.VideoCapture(self.cap_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("ERROR: Camera failed to initialize!")
            return
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
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
        self.IS_READING_MODE_RUNNING = False
        self.close_hotkeys = [ ord("q"), ord("Q") ]
        self.caption_hotkeys = [ ord("c"), ord("C") ]
        self.reading_mode_hotkeys = [ ord("r"), ord("R") ]
        self.ocr_image: npt.NDArray | None = None

        self.warnings: list[str] = []

        self.__run()
        
        print("\n========================")
        print("-----iVision Closed-----")
        print("========================\n")

    def __get_models(self):
        print("\n-----Loading Models-----\n")

        print("-----Loading YOLO-----")
        self.model_object_detector = ObjectDetectorTFLite()
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

        print("-----Loading Grid Obstacle Detector-----")
        self.grid_obstacle_detector = GridObstacleDetector(self.width, self.height)
        
        print("-----Loading Grid Obstacle Detector-----")
        self.grid_warning_system = GridWarningSystem(self.width, self.height)
        
        print("-----Loading Complexity Estimation-----")
        self.model_complexity_estimator = ComplexityEstimator()
        
        print("-----Loading Scene Classifier-----")
        self.model_scene_classifier = SceneClassifier()

        print("-----Loading Scene Narration-----")
        self.model_scene_narrator = SceneNarrator()

        print("-----Loading Reading Mode-----")
        self.reading_mode = ReadingMode()
        
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
        depth_bw_bgr = output_image.copy()

        yolo_centroids = self.model_object_detector.draw_objects_with_depth(
            output_image, depth_bw, True, self.depth_warning_threshold
        )
        yolo_depth_image = output_image.copy()
        
        self.grid_obstacle_detector.draw_grid(
            output_image, depth_bw, boxes, masks, self.depth_warning_threshold, True
        )
        depth_centroids = self.grid_obstacle_detector.get_obstacle_centroids()
        grid_depth_image = output_image.copy()

        warning_image = frame.copy()
        self.grid_warning_system.draw_grid(warning_image, frame, yolo_centroids, depth_centroids)
        self.warnings = self.grid_warning_system.get_warnings()

        if not self.side_by_side:
            return output_image

        yolo_image = frame.copy()
        self.model_object_detector.draw_objects(yolo_image)

        label = f"{complexity.name}, {scene_type.name}, {weather.name}"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # output_image = np.vstack((np.hstack((frame, yolo_image, depth_bw_bgr)), 
                                #   np.hstack((yolo_depth_image, grid_depth_image, warning_image))))
        
        if self.ocr_image is None:
            # output_image = np.vstack((np.hstack((frame, frame)), 
            #                           np.hstack((grid_depth_image, warning_image))))
            output_image = np.vstack((np.hstack((frame, yolo_image, depth_bw_bgr)), 
                                      np.hstack((grid_depth_image, warning_image, frame))))

        else:
            # output_image = np.vstack((np.hstack((frame, self.ocr_image)), 
            #                           np.hstack((grid_depth_image, warning_image))))
            output_image = np.vstack((np.hstack((frame, yolo_image, depth_bw_bgr)), 
                                      np.hstack((grid_depth_image, warning_image, self.ocr_image))))


        return output_image

    def __send_warnings(self):
        "Placeholder function for when a STT Warning System is added."
        pass

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
        
        SLEEP_SECS = 0.25
        WINDOW_SIZE = 10
        prev_weather = Weather.BRIGHT
        weather_window = SlidingWindow(WINDOW_SIZE, prev_weather)
        scene_type_map = {"indoor": SceneType.INDOOR, "outdoor": SceneType.OUTDOOR}
        
        prev_frame = np.zeros(1)

        while self.IS_RUNNING:
            with self.frame_lock:
                frame = self.frame.copy()
            
            if np.array_equal(frame, prev_frame):
                time.sleep(0)
                continue
            prev_frame = frame

            scene_type = scene_type_map[self.model_scene_classifier.get_scene_type_binary(frame)]
            complexity, weather, confidence = self.model_complexity_estimator.predict(scene_type, frame)

            weather_window.append(weather)
            max_weather = weather_window.get_max_val()

            if max_weather != prev_weather:
                prev_weather = max_weather
                max_scene_type = self.model_complexity_estimator.weather_to_scene[max_weather]
                max_complexity = self.model_complexity_estimator.weather_to_complexity[max_weather]
                
                cur_datetime = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                print((f"\n"
                       f"{cur_datetime} | Complexity: {max_complexity.name}\n"
                       f"{cur_datetime} | Scene: {max_scene_type.name}\n"
                       f"{cur_datetime} | Weather: {max_weather.name}\n"))
                
                with self.complexity_lock:
                    self.complexity = max_complexity
                    self.scene_type = max_scene_type
                    self.weather = max_weather

            time.sleep(SLEEP_SECS)
        
        print("-----Stopped Complexity Estimation Thread-----")

    def __captioning_thread(self, frame):
        print("\n-----STARTING NARRATION-----")
        narration_text = self.model_scene_narrator.get_narration(frame, self.model_scene_classifier)
        print(f"Final Narration: {narration_text}")
        print("-----STOPPING NARRATION-----\n")
        self.IS_CAPTION_RUNNING = False

    def __reading_mode_thread(self, frame: npt.NDArray):
        print("\n-----STARTING OCR-----")
        texts, scores, boxes = self.reading_mode.get_text(frame)
        print(f"Detected {len(texts)} texts!")
        self.ocr_image = self.reading_mode.draw_boxes(frame)
        
        print("-----PRINTING DETECTED TEXT-----")
        for text, score in zip(texts, scores):
            print(f"Text ({score:0.2f}): \"{text}\"")
        print("--------PRINTING STOPPED--------")

        print("-----STOPPING OCR-----\n")
        self.IS_READING_MODE_RUNNING = False

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
            
            output_image = self.__get_output_image(frame.copy())
            cv2.imshow(f"iVision", output_image)
            self.__send_warnings()
            
            key_pressed = cv2.waitKey(1)

            if key_pressed in self.close_hotkeys:
                break
            elif not self.IS_CAPTION_RUNNING and key_pressed in self.caption_hotkeys:
                self.IS_CAPTION_RUNNING = True
                threading.Thread(target=self.__captioning_thread, args=(frame,), daemon=True).start()
            elif not self.IS_READING_MODE_RUNNING and key_pressed in self.reading_mode_hotkeys:
                self.IS_READING_MODE_RUNNING = True
                threading.Thread(target=self.__reading_mode_thread, args=(frame,), daemon=True).start()
            elif not self.IS_READING_MODE_RUNNING and (keyboard.is_pressed('r') or keyboard.is_pressed('R')):
                self.IS_READING_MODE_RUNNING = True
                threading.Thread(target=self.__reading_mode_thread, args=(frame,), daemon=True).start()
            
        print("-----Stopping Main Thread-----")
        self.IS_RUNNING = False
        camera_thread.join()
        complexity_estimation_thread.join()
        cv2.destroyAllWindows()
        print("\n-----Stopped Main Thread-----\n")

if __name__ == "__main__":
    iVision(
        model_yolo_type=(ObjectDetectorTFLite.ModelType.YOLO_SEGMENT, ObjectDetectorTFLite.ModelType.YOLO_SEGMENT),
        model_depth_type=(DepthEstimator.ModelType.MIDAS_V21, DepthEstimator.ModelType.DEPTH_ANYTHING_V2),
        depth_warning_threshold=180,
        side_by_side=True,
        debug=False,
        cap_id=0
    )
