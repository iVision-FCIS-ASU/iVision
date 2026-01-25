import torch
import numpy as np
from typing import Literal
from midas.model_loader import model_paths, load_model
from midas.model_processor import process, create_side_by_side

class MiDaS:
    def __init__(self, model_type: Literal["midas_v21_small_256", "dpt_swin2_tiny_256"]):
        self.model_type = model_type
        model_path = model_paths[self.model_type]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, model_path, model_type, False)
    
    def get_depth_image(self, frame):
        with torch.no_grad():
            original_image_rgb = np.flip(frame, 2)
            image = self.transform({"image": original_image_rgb/255})["image"]

            prediction = process(self.device, self.model, self.model_type, 
                                image, (self.net_w, self.net_h),
                                original_image_rgb.shape[1::-1], False, True)
            
            return create_side_by_side(None, prediction, False)