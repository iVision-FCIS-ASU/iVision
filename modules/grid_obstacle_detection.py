import cv2
import numpy as np
from itertools import pairwise
from numpy import typing as npt
from scipy.cluster.hierarchy import DisjointSet
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

class GridObstacleDetector:
    def __init__(
        self, 
        w: int, 
        h: int, 
        grid_size: tuple[int, int] = (11, 11),
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

        self.color = color
        self.grid_thickness = grid_thickness

        self.overlay_color_ratio = overlay_color_ratio
        self.inverse_overlay_color_ratio = 1 - overlay_color_ratio
        self.overlay_color = overlay_color_ratio * np.array(self.color, dtype=np.uint8)
        self.depth_intensity_threshold = depth_warning_threshold
        self.depth_mean_threshold = depth_warning_threshold / 255
        self.depth_percentage_threshold = depth_percentage_threshold

        self.obstacle_centroid_radius = obstacle_centroid_radius
        self.obstacle_thickness = obstacle_thickness

        self.__create_grid_points()
        self.grid_set = DisjointSet()

        self.polygon_centroids: None | list[tuple[int, int]] = None

    def __create_grid_points(self):
        grid_size_x, grid_size_y = self.grid_size
        grid_x_loop = grid_size_x // 2
        grid_y_loop = grid_size_y // 2

        w_left = self.w // grid_size_x
        w_right = self.w - w_left
        h_up = self.h // grid_size_y
        h_down = self.h - h_up

        x_points = [0, self.w]
        y_points = [0, self.h]

        for i in range(grid_x_loop):
            x = w_left + i * w_left
            x_points.append(x)

        for i in range(grid_x_loop):
            x = w_right - i * w_left
            x_points.append(x)

        for i in range(grid_y_loop):
            y = h_up + i * h_up
            y_points.append(y)

        for i in range(grid_y_loop):
            y = h_down - i * h_up
            y_points.append(y)

        self.x_points: npt.NDArray[np.integer] = np.sort(x_points)
        self.y_points: npt.NDArray[np.integer] = np.sort(y_points)

        self.x_points_pos: dict[int, int] = { val: idx for idx, val in enumerate(self.x_points) }
        self.y_points_pos: dict[int, int] = { val: idx for idx, val in enumerate(self.y_points) }
        
        self.grids = { (j, i): (xi, yi, xj, yj) for j, (yi, yj) in enumerate(pairwise(self.y_points)) for i, (xi, xj) in enumerate(pairwise(self.x_points)) }
        self.grids_inverse = { grid: pos for pos, grid in self.grids.items() }

    def __get_obstacle_grids_mean(self, depth_bw: npt.NDArray, output_img: npt.NDArray, grid: tuple[int, int, int, int]):
        xi, yi, xj, yj = grid
        depth_mean = np.mean(depth_bw[yi:yj, xi:xj]) / 255
        
        if depth_mean > self.depth_mean_threshold:
            overlay_color = depth_mean * self.overlay_color
            output_img[yi:yj, xi:xj, :] = self.inverse_overlay_color_ratio * output_img[yi:yj, xi:xj, :] + overlay_color
            self.grid_set.add(self.grids_inverse[grid])

    def __get_obstacle_grids_percentage(self, depth_bw: npt.NDArray, output_img: npt.NDArray, grid: tuple[int, int, int, int]):
        xi, yi, xj, yj = grid
        depth_percentage = np.mean(depth_bw[yi:yj, xi:xj] > self.depth_intensity_threshold)

        if depth_percentage > self.depth_percentage_threshold:
            overlay_color = depth_percentage * self.overlay_color
            output_img[yi:yj, xi:xj, :] = self.inverse_overlay_color_ratio * output_img[yi:yj, xi:xj, :] + overlay_color
            self.grid_set.add(self.grids_inverse[grid])

    def __draw_overlays(self, depth_bw: npt.NDArray, output_img: npt.NDArray):
        self.grid_set = DisjointSet()

        for grid in self.grids.values():
            # self.__get_obstacle_grids_mean(depth_bw, output_img, grid)
            self.__get_obstacle_grids_percentage(depth_bw, output_img, grid)

    def __draw_gridlines(self, output_img: npt.NDArray):
        for x in self.x_points[1:-1]:
            cv2.line(output_img, (x, 0), (x, self.h), color=self.color, thickness=self.grid_thickness)

        for y in self.y_points[1:-1]:
            cv2.line(output_img, (0, y), (self.w, y), color=self.color, thickness=self.grid_thickness)

    def __connect_adjacent_grids(self):
        for grid_pos in self.grid_set:
            j, i = grid_pos
            left  = (j, i - 1)
            right = (j, i + 1)
            above = (j - 1, i)
            below = (j + 1, i)

            if left in self.grid_set:
                self.grid_set.merge(grid_pos, left)
            if right in self.grid_set:
                self.grid_set.merge(grid_pos, right)
            if above in self.grid_set:
                self.grid_set.merge(grid_pos, above)
            if below in self.grid_set:
                self.grid_set.merge(grid_pos, below)
        
    def __union_connected_grids(self) -> list[Polygon]:
        polygons: list[Polygon] = []
        
        for subset in self.grid_set.subsets():
            grids = [box(*self.grids[grid_index]) for grid_index in subset]
            polygons.append(unary_union(grids).simplify(0))

        return polygons

    def __draw_connected_grids(self, output_img: npt.NDArray):
        self.__connect_adjacent_grids()
        polygons = self.__union_connected_grids()

        self.polygon_centroids = [(int(polygon.centroid.x), int(polygon.centroid.y)) for polygon in polygons]
        polygon_coords = [np.array(polygon.exterior.coords, dtype=np.int32).reshape((-1, 1, 2)) for polygon in polygons]

        for centroid, coords in zip(self.polygon_centroids, polygon_coords):
            cv2.circle(output_img, centroid, radius=self.obstacle_centroid_radius, color=self.color, thickness=-1)
            cv2.polylines(output_img, [coords], isClosed=False, color=self.color, thickness=self.obstacle_thickness)

    def get_obstacle_centroids(self) -> None | list[tuple[int, int]]:
        return self.polygon_centroids

    def draw_grid(
        self, 
        output_img: npt.NDArray, 
        depth_bw: npt.NDArray,
        masks: list[npt.NDArray] = [],
        depth_warning_threshold: int = 150,
        draw_gridlines = False
    ) -> npt.NDArray:
        self.depth_intensity_threshold = depth_warning_threshold
        self.depth_mean_threshold = depth_warning_threshold / 255
        
        depth_bw_without_objects = depth_bw.copy()
        for mask in masks:
            depth_bw_without_objects[mask] = 0

        self.__draw_overlays(depth_bw_without_objects, output_img)
        if draw_gridlines:
            self.__draw_gridlines(output_img)
        self.__draw_connected_grids(output_img)
