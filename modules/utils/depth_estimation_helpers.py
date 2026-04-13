import numpy as np
from numpy import typing as npt

def get_object_depth_box(depth_bw: npt.NDArray, box: tuple[int, int, int, int, int, float]) -> tuple[int, int, int]:
    x1, y1, x2, y2, _, _ = box
    region = depth_bw[y1:y2, x1:x2].flatten()
    min_depth = int(np.min(region))
    max_depth = int(np.max(region))

    k = int(0.50 * len(region))
    top = np.partition(region, -k)[-k:]
    depth = int(np.mean(top))
    return depth, min_depth, max_depth

def get_object_depth_mask(depth_bw: npt.NDArray, mask: npt.NDArray) -> tuple[int, int, int]:
    min_depth = int(np.min(depth_bw[mask]))
    max_depth = int(np.max(depth_bw[mask]))
    mean_depth = int(np.mean(depth_bw[mask]))
    return mean_depth, min_depth, max_depth
