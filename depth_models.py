import cv2
import torch
import numpy as np
import onnxruntime as ort
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

class DepthAnythingV2:
    def __init__(self):
        self.sess = ort.InferenceSession(
        "weights/depth_anything_v2_vits.onnx",
        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    @staticmethod
    def __preprocess(frame, target_size=518):
        """Resize to square, normalize, convert to NCHW float32"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # pad to square (center)
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        # normalize
        padded = padded.astype(np.float32) / 255.0
        padded = padded.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
        return padded, (h, w), (top, bottom, left, right)

    @staticmethod
    def __postprocess(depth, orig_shape, pad_info, target_size=518):
        """Remove padding and resize back to original aspect ratio"""
        top, bottom, left, right = pad_info
        depth = depth[top:target_size-bottom, left:target_size-right]
        depth = cv2.resize(depth, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)
        return depth

    def get_depth_image(self, frame):
        inp, orig_shape, pad_info = self.__preprocess(frame)

        # Run ONNX inference
        outputs = self.sess.run(None, {"input": inp})
        # print("Output shape:", outputs[0].shape)
        # depth_raw = outputs[0][0, 0]  # shape (H, W)

        out = outputs[0]               # ONNX outputs list -> first output
        # collapse singleton dims so we end up with a 2D HxW depth map
        depth_raw = np.squeeze(out)    # works for (1,1,H,W), (1,H,W), (H,W), (1,H*W) if square-ish

        # sanity check
        if depth_raw.ndim != 2:
            # fallback: try reshaping flattened vector into square
            if depth_raw.ndim == 1:
                side = int(np.sqrt(depth_raw.shape[0]))
                if side * side == depth_raw.shape[0]:
                    depth_raw = depth_raw.reshape(side, side)
                else:
                    raise ValueError(f"Can't reshape flattened output of size {depth_raw.shape[0]} into square.")
            else:
                raise ValueError(f"Unexpected depth map shape after squeeze: {depth_raw.shape}")


        # Postprocess: remove padding + restore original shape
        depth_resized = self.__postprocess(depth_raw, orig_shape, pad_info)

        # Normalize for display
        depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # return depth_colored
        return depth_norm, depth_colored
