import numpy as np
from numpy import typing as npt

# def get_mean_depth_box(depth_image, box, mask_model):
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     mean_depth = int(np.mean(depth_image[y1:y2, x1:x2]))
#     min_depth = int(np.min(depth_image[y1:y2, x1:x2]))
#     max_depth = int(np.max(depth_image[y1:y2, x1:x2]))
#     return mean_depth, min_depth, max_depth

# def get_mean_depth_mask(depth_image, mask, mask_model):
#     mask = mask_model[0].cpu().numpy() > 0.5
#     mean_depth = int(np.mean(depth_image[mask]))
#     min_depth = int(np.min(depth_image[mask]))
#     max_depth = int(np.max(depth_image[mask]))
#     return mean_depth, min_depth, max_depth

def get_mean_depth_box(depth_bw: npt.NDArray, box: tuple[int, int, int, int, int, float]) -> tuple[int, int, int]:
    x1, y1, x2, y2, _, _ = box
    mean_depth = int(np.mean(depth_bw[y1:y2, x1:x2]))
    min_depth = int(np.min(depth_bw[y1:y2, x1:x2]))
    max_depth = int(np.max(depth_bw[y1:y2, x1:x2]))
    return mean_depth, min_depth, max_depth

def get_mean_depth_mask(depth_bw: npt.NDArray, mask: npt.NDArray) -> tuple[int, int, int]:
    mean_depth = int(np.mean(depth_bw[mask]))
    min_depth = int(np.min(depth_bw[mask]))
    max_depth = int(np.max(depth_bw[mask]))
    return mean_depth, min_depth, max_depth
