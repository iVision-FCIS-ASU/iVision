import cv2
import numpy as np
import numpy.typing as npt
import line_profiler
from modules.complexity_estimation import ComplexityEstimator, Complexity, SceneType, Weather
from modules.depth_estimation import DepthEstimator
from modules.object_detection_tflite import ObjectDetectorTFLite
from modules.scene_classification import SceneClassifier

model_complexity = ComplexityEstimator()
model_scene = SceneClassifier()

def get_simple_narration(
    boxes: list[tuple[int, int, int, int, int, float]], 
    centroids: list[tuple[tuple[int, int], str]], 
    depths: list[int],
    frame: npt.NDArray
):
    scene_binary, scene_type = model_scene.get_scene_type(frame)
    scene_type = scene_type.replace("_", " ")
    
    scene_type_binary_map = {"indoor": SceneType.INDOOR, "outdoor": SceneType.OUTDOOR}
    scene_type_binary = scene_type_binary_map[scene_binary]
    complexity, weather, conf = model_complexity.predict(scene_type_binary, frame)

    caption = f"You are {scene_binary} in a {scene_type}. "
    if scene_binary:
        caption += f"The place is {weather.name.lower()}. "
    elif weather is not Weather.NIGHT:
        caption += f"The weather is {weather.name.lower()}. "
    else:
        caption += f"It's night-time. "

    foreground_thresh = 0.66
    midground_thresh = 0.33
    foreground: dict[str, int] = {}
    midground: dict[str, int] = {}
    background: dict[str, int] = {}

    for box, centroid, depth in zip(boxes, centroids, depths):
        x1, y1, x2, y2, _, _ = box
        (x, y), cls_name = centroid
        depth = depth / 255.0

        if depth > foreground_thresh:
            if cls_name not in foreground:
                foreground[cls_name] = 0
            foreground[cls_name] += 1
        elif depth > midground_thresh:
            if cls_name not in midground:
                midground[cls_name] = 0
            midground[cls_name] += 1
        else:
            if cls_name not in background:
                background[cls_name] = 0
            background[cls_name] += 1

    def get_objects_str(d: dict[str, int]):
        caption = ""
        single_objects = set()
        multi_objects = set()

        for name, count in d.items():
            if count == 1:
                single_objects.add(name)
            else:
                multi_objects.add(name)
        
        vowels = { "a", "e", "o", "u", "i" }

        if len(single_objects):
            caption += "is "
        
        for idx, name in enumerate(single_objects):
            if name[0] in vowels:
                caption += f"an {name}, "
            else:
                caption += f"a {name}, "
            if idx == len(single_objects) - 2:
                caption = caption[:-2] + "and "
        
        if len(single_objects):
            caption = caption[:-2] + ". "
            if len(multi_objects):
                caption += "Also, there are several "
        else:
            caption += "are several "
        
        for idx, name in enumerate(multi_objects):
            caption += f"{name}, "

            if idx == len(multi_objects) - 2:
                caption = caption[:-2] + "and "
        
        caption = caption[:-2] + ". "
        return caption

    
    foreground_count = sum(foreground.values())
    if foreground_count != 0:
        caption += "In the foreground, there "
        caption += get_objects_str(foreground)

    midground_count = sum(midground.values())
    if midground_count != 0:
        caption += "In the midground, there "
        caption += get_objects_str(midground)

    background_count = sum(background.values())
    if background_count != 0:
        caption += "In the background, there "
        caption += get_objects_str(background)

    print(caption)

@line_profiler.profile
def run_simple_narration():
    model_yolo = ObjectDetectorTFLite()
    model_depth = DepthEstimator()

    yolo_model_type = ObjectDetectorTFLite.ModelType.YOLO_SEGMENT
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break

        depth_bw, _ = model_depth.get_depth_image(frame, DepthEstimator.ModelType.DEPTH_ANYTHING_V2)
        output_image = frame.copy()
        
        boxes, masks, centroids = model_yolo.get_objects(frame, yolo_model_type)
        depths = model_yolo.get_object_depths(depth_bw)
        _ = model_yolo.draw_objects_with_depth(output_image, depth_bw, draw_masks=True)
    
        cv2.imshow(f"Simple Narration Test", output_image)

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord("q") or pressed_key == ord("Q"):
            break
        elif pressed_key == ord("c") or pressed_key == ord("C"):
            get_simple_narration(boxes, centroids, depths, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_simple_narration()
