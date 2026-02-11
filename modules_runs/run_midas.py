import torch
import cv2
import time
import numpy as np
import line_profiler
from typing import Literal
from imutils.video import VideoStream

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.midas.model_loader import model_paths, load_model
from modules.midas.model_processor import process, create_side_by_side

@line_profiler.profile
def run(
        model_type: Literal["midas_v21_small_256", "dpt_swin2_tiny_256"], 
        optimize: bool,
        side: bool, 
        grayscale: bool
    ):

    model_path = model_paths[model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize)

    with torch.inference_mode():
        fps = 1
        video = VideoStream(0).start()
        time_start = time.time()
        frame_index = 0
        
        while True:
            frame = video.read()
            if frame is None:
                continue
            
            original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
            image = transform({"image": original_image_rgb/255})["image"]

            prediction = process(device, model, model_type, image, (net_w, net_h),
                                    original_image_rgb.shape[1::-1], optimize, True)

            original_image_bgr = np.flip(original_image_rgb, 2) if side else None
            depth_bw, content = create_side_by_side(original_image_bgr, prediction, grayscale)
            cv2.imshow('MiDaS Depth Estimation - Press q to close window ', content)

            alpha = 0.1
            if time.time()-time_start > 0:
                fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
                time_start = time.time()
            print(f"\rFPS: {round(fps,2)}", end="")

            pressed_key = cv2.waitKey(1)
            if pressed_key == ord("q") or pressed_key == ord("Q"):  
                break

            frame_index += 1
        
    print("\nFinished")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    run(
        model_type="midas_v21_small_256",
        optimize=False,
        side=False,
        grayscale=True
    )
