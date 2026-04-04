from __future__ import annotations
import cv2
import numpy as np
import line_profiler
from enum import Enum, auto
from itertools import pairwise
from numpy import typing as npt
from .utils.sliding_window import SlidingWindow

class GridWarningSystem:
    class Status(Enum):
        CLEAR = auto()
        OBSTRUCTED = auto()

    def __init__(
        self, 
        w: int, 
        h: int, 
        grid_size: tuple[int, int] = (11, 3),
        grid_ratio: tuple[float, float] = (0.6, 0.6),
        grid_window_size: int = 10,
        color: tuple[int, int, int] = (0, 0, 255),
        grid_thickness: int = 1,
        overlay_color_ratio: float = 0.5,
        depth_warning_threshold: int = 150,
        depth_percentage_threshold: float = 0.25,
        obstacle_centroid_radius: int = 10,
        obstacle_thickness: int = 3,
    ):
        self.w = w
        self.h = h
        self.grid_size = grid_size
        self.grid_ratio = grid_ratio
        self.grid_window_size = grid_window_size

        self.color = color
        self.grid_thickness = grid_thickness

        self.overlay_color_ratio = overlay_color_ratio
        self.inverse_overlay_color_ratio = 1 - overlay_color_ratio
        self.overlay_color = overlay_color_ratio * np.array(self.color, dtype=np.uint8)
        self.depth_warning_threshold = depth_warning_threshold
        self.depth_mean_threshold = depth_warning_threshold / 255
        self.depth_percentage_threshold = depth_percentage_threshold

        self.obstacle_centroid_radius = obstacle_centroid_radius
        self.obstacle_thickness = obstacle_thickness

        self.__create_grids()
    
    def __create_grids(self):
        grid_size_x, grid_size_y = self.grid_size
        grid_ratio_x, grid_ratio_y = self.grid_ratio
        grid_x_loop = grid_size_x // 2
        grid_y_loop = grid_size_y // 2

        w_left = self.w // grid_size_x
        w_right = self.w - w_left
        h_up = int(grid_ratio_y * (self.h // grid_size_y))
        h_down = self.h - h_up

        x_points = [0, self.w]
        y_points = [0, h_up, h_down, self.h]

        grid_x_start = int(np.ceil(grid_x_loop * grid_ratio_x)) - 1

        for i in range(grid_x_start, grid_x_loop):
            x = w_left + i * w_left
            x_points.append(x)

        for i in range(grid_x_start, grid_x_loop):
            x = w_right - i * w_left
            x_points.append(x)

        self.x_points: npt.NDArray[np.integer] = np.sort(x_points)
        self.y_points: npt.NDArray[np.integer] = np.sort(y_points)

        self.x_points_pos: dict[int, int] = { val: idx for idx, val in enumerate(self.x_points) }
        self.y_points_pos: dict[int, int] = { val: idx for idx, val in enumerate(self.y_points) }
        
        self.grids: dict[tuple[int, int], tuple[int, int, int, int]] = { 
            (j, i): (xi, yi, xj, yj) 
            for j, (yi, yj) in enumerate(pairwise(self.y_points)) 
            for i, (xi, xj) in enumerate(pairwise(self.x_points))
        }
        self.grids_inverse = { grid: pos for pos, grid in self.grids.items() }
        
        self.grid_windows = { pos: SlidingWindow(self.grid_window_size, self.Status.CLEAR) for pos in self.grids }

        for j in range(0, 3):
            shared_window = SlidingWindow(self.grid_window_size, self.Status.CLEAR)
            for i in range(1, 6):
                self.grid_windows[(j, i)] = shared_window

        self.grid_warning_data: dict[tuple[int, int], list[tuple[tuple[int, int], str]]] = {}
        self.grid_warning_message: dict[tuple[int, int], str] = {
            (0, 0): "left, head level",
            (1, 0): "left, chest level",
            (2, 0): "left, leg level",
            (0, 1): "front left, head level",
            (1, 1): "front left, chest level",
            (2, 1): "front left, leg level",
            (0, 2): "front slightly left, head level",
            (1, 2): "front slightly left, chest level",
            (2, 2): "front slightly left, leg level",
            (0, 3): "front, head level",
            (1, 3): "front, chest level",
            (2, 3): "front, leg level",
            (0, 4): "front slightly right, head level",
            (1, 4): "front slightly right, chest level",
            (2, 4): "front slightly right, leg level",
            (0, 5): "front right, head level",
            (1, 5): "front right, chest level",
            (2, 5): "front right, leg level",
            (0, 6): "right, head level",
            (1, 6): "right, chest level",
            (2, 6): "right, leg level",
        }

    def __collect_obstacles_data(self, centroids: list[tuple[tuple[int, int], str]]):
        self.grid_warning_data = {}

        for centroid, cls in centroids:
            x, y = centroid
            
            for pos, grid in self.grids.items():
                xi, yi, xj, yj = grid
                
                if (xi <= x < xj) and (yi <= y < yj):
                    if pos not in self.grid_warning_data:
                        self.grid_warning_data[pos] = []
                    self.grid_warning_data[pos].append((centroid, cls))

    def __draw_overlays(self, output_img: npt.NDArray):
        max_count = 0
        for grid_warning_list in self.grid_warning_data.values():
            max_count = max(max_count, len(grid_warning_list))
        max_count /= self.overlay_color_ratio

        for pos, centroids in self.grid_warning_data.items():
            xi, yi, xj, yj = self.grids[pos]
            overlay_color = (len(centroids) / max_count) * self.overlay_color
            output_img[yi:yj, xi:xj, :] = self.inverse_overlay_color_ratio * output_img[yi:yj, xi:xj, :] + overlay_color

    def __draw_gridlines(self, output_img: npt.NDArray):
        for x in self.x_points[1:-1]:
            cv2.line(output_img, (x, 0), (x, self.h), color=self.color, thickness=self.grid_thickness)

        for y in self.y_points[1:-1]:
            cv2.line(output_img, (0, y), (self.w, y), color=self.color, thickness=self.grid_thickness)
  
    def __draw_centroids(self, output_img: npt.NDArray, centroids: list[tuple[tuple[int, int], str]]):
        for centroid, cls in centroids:
            cv2.circle(output_img, centroid, radius=self.obstacle_centroid_radius, color=self.color, thickness=-1)
            text_size, _ = cv2.getTextSize(cls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            x = int(centroid[0] - text_size[0]/2)
            y = int(centroid[1] - text_size[1]/2) - 5
            cv2.putText(output_img, cls, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
    
    def __update_windows(self):
        # for pos in self.grids:
        #     if pos in self.grid_warning_data:
        #         self.grid_windows[pos].append(self.Status.OBSTRUCTED)
        #     else:
        #         self.grid_windows[pos].append(self.Status.CLEAR)

        individual_grids = [ (0, 0), (1, 0), (2, 0), (0, 6), (1, 6), (2, 6) ]

        shared_grids = [
            [ (0, 1), (0, 2), (0, 3), (0, 4), (0, 5) ],
            [ (1, 1), (1, 2), (1, 3), (1, 4), (1, 5) ],
            [ (2, 1), (2, 2), (2, 3), (2, 4), (2, 5) ]
        ]

        for pos in individual_grids:
            if pos in self.grid_warning_data:
                self.grid_windows[pos].append(self.Status.OBSTRUCTED)
            else:
                self.grid_windows[pos].append(self.Status.CLEAR)
        
        for shared_grid in shared_grids:
            shared_status = self.Status.CLEAR
            for pos in shared_grid:
                if pos in self.grid_warning_data:
                    shared_status = self.Status.OBSTRUCTED
                    break
            self.grid_windows[shared_grid[0]].append(shared_status)

    def __create_warnings(self):
        self.warnings: list[str] = []
        for pos, centroids in self.grid_warning_data.items():
            if self.grid_windows[pos].get_max_val() == self.Status.CLEAR:
                continue

            for _, cls in centroids:
                self.warnings.append(f"{cls}, {self.grid_warning_message[pos]}")

    def get_warnings(self) -> list[str]:
        return self.warnings
    
    def draw_grid(
        self, 
        output_img: npt.NDArray, 
        frame: npt.NDArray, 
        yolo_centroids: list[tuple[tuple[int, int], str]], 
        depth_centroids: list[tuple[int, int]]
    ) -> None:
        centroids = yolo_centroids + [((centroid), "unknown") for centroid in depth_centroids]
        
        self.__collect_obstacles_data(centroids)
        self.__draw_overlays(output_img)
        self.__draw_gridlines(output_img)
        self.__draw_centroids(output_img, centroids)
        self.__update_windows()
        self.__create_warnings()
    