import cv2
import requests
import numpy as np
import threading
import line_profiler
from numpy import typing as npt
from scene_classification import SceneClassifier
from scene_narration import SceneNarrator

IS_CAPTION_RUNNING = False

def get_frame(url: None | str) -> None | npt.NDArray:
    if url is None:
        return None

    response = requests.get(url)
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def narrate(scene_narrator: SceneNarrator, frame: npt.NDArray, scene_classifier: SceneClassifier):
    caption = scene_narrator.get_narration(frame, scene_classifier)
    print(caption)
    global IS_CAPTION_RUNNING
    IS_CAPTION_RUNNING = False

@line_profiler.profile
def run_scene_narration(frame: None | npt.NDArray):
    scene_classifier = SceneClassifier()
    scene_narrator = SceneNarrator()

    if frame is not None:
        caption = narrate(scene_narrator, frame, scene_classifier)
        print(caption)
        return

    global IS_CAPTION_RUNNING
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture")
            break
        
        cv2.imshow("Scene Narration", frame)

        key_pressed = cv2.waitKey(1)
        if not IS_CAPTION_RUNNING and (key_pressed == ord("c") or key_pressed == ord("C")):
            IS_CAPTION_RUNNING = True
            threading.Thread(target=narrate, args=(scene_narrator, frame, scene_classifier), daemon=True).start()

        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = None
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # url = "https://thumbs.dreamstime.com/b/cute-cat-sleeping-street-car-random-58655731.jpg"
    frame = get_frame(url)

    run_scene_narration(frame)
