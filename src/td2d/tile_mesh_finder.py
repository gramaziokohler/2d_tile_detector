import argparse
import threading
from dataclasses import dataclass
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import cv2 as cv
import numpy
from compas.datastructures import Mesh
from compas.datastructures import mesh_thicken
from compas.geometry import Polyhedron
from compas.data import json_dump

from td2d.gui import UiManager
from td2d.gui import UserInput
from td2d.device import SingleImageDevice
from td2d.genicam_device import GenTlDevice


class ThreadWithReturn(threading.Thread):
    """
    TODO: for potential future use (parallelizing tile mesh creation)
    Lets you collect the result of a threaded operation once the thread is finished
    >>> func = lambda x,y : (x, y)
    >>> arg1 = 1
    >>> arg2 = 2
    >>> t = ThreadWithReturn(target=func, args=[arg1, arg2])
    >>> t.start()
    >>> t.join()
    >>> t.result
    (1, 2)
    """

    def __init__(self, target: Callable, args: List[Any]):
        super(ThreadWithReturn, self).__init__()
        self.callable = target
        self.args = args
        self.is_running = False
        self._result = None

    def run(self) -> None:
        self.is_running = True
        self._result = self.callable(*self.args)
        self.is_running = False

    @property
    def result(self) -> Any:
        """
        Get the result of the operation performed by `target`
        :return:
        """
        if self.is_running:
            raise AssertionError("Thread has not yet finished or never ran!")
        return self._result


class TileFinder:
    """
    Finds tiles in an image, created meshed and serializes them to a file

    """

    DEFAULT_TILE_THICKNESS = 2

    def __init__(self, capture_device, output_file):
        self.is_running = False
        self.capture_device = capture_device
        self.output_file = output_file
        self.ui_manager = UiManager("Display")
        self.ui_manager.THRESHOLD_SLIDER_MAX = 255

    def run(self) -> None:
        """
        Start the finder.

        Until finder.stop() is called, do:
        1. get image from capture device
        2. read search parameters from user input
        3. find contours
        4. draw contours on top of the image
        5. if asked to save, create polygons from contours -> create thick meshes from polygons
            6. serialize meshes to file
        """

        self.is_running = True
        self.ui_manager.start()
        while self.is_running:
            image = self.capture_device.get_next_image()
            user_input = self.ui_manager.get_user_input()
            tiles = self.find_tiles(image, user_input.threshold, user_input.min_area, user_input.max_area)
            simplified_tiles = self.simplified_tiles(tiles)
            self.ui_manager.draw_selected_tiles(image, simplified_tiles)

            if user_input.should_save:
                print("started creating meshes..")
                meshes = self.create_meshes(simplified_tiles)
                print("started creating meshes..Done")
                self.save_meshes(meshes)
            if user_input.should_exit:
                self.stop()

    def stop(self) -> None:
        """Stop the finder"""
        self.is_running = False

    @staticmethod
    def simplified_tiles(tiles):
        return tuple(cv.convexHull(tile) for tile in tiles)

    @staticmethod
    def find_tiles(image: numpy.ndarray, threshold: int, min_area: int, max_area: int) -> Tuple[numpy.ndarray]:
        """
        Identify tiles in image by finding their shape's contour.
        Filters out contours whose area is smaller than min_area or larger than max_area.
        Threshold is the pixel intensity value used as the thresholding value when creating a binary image.

        :param image: an image of tiles
        :param threshold: thresholding value
        :param min_area: max tile area allowed
        :param max_area: min tile area allowed
        :return: Tuple containing all the found contours which comply with the filter values
        """
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(gray_img, threshold, 255, 0)
        contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        def filter_(contour):
            area = cv.contourArea(contour)
            return min_area < area < max_area

        return tuple(cntr for cntr in contours if filter_(cntr))

    def save_meshes(self, meshes: List[Polyhedron]) -> None:
        """
        Serialize the list of meshes to json file
        :param meshes: list of meshes
        """
        json_dump(meshes, self.output_file)

    def create_meshes(self, contours, thickness=DEFAULT_TILE_THICKNESS) -> List[Polyhedron]:
        """
        Iterate on the tule of contours and generate inflated (thickened/extruded) meshes from them
        :param contours: the contours tuple as found by OpenCV
        :param thickness: the thickness value to use when extruding
        :return: list of meshes
        """
        tiles = []

        for cnt in contours:
            self._remove_weird_extra_dimension_from_numpy_array(cnt)
            vertices = [[v[0], v[1], 0] for v in cnt]
            face = [list(range(len(vertices)))]

            mesh = Mesh.from_vertices_and_faces(vertices, face)
            mesh = mesh_thicken(mesh, thickness=thickness)
            vertices, faces = mesh.to_vertices_and_faces(triangulated=True)
            tiles.append(Polyhedron(vertices, faces))
        return tiles

    @staticmethod
    def _remove_weird_extra_dimension_from_numpy_array(contour):
        """(m, 1, n) => (m, n)"""
        num_of_points = contour.shape[0]
        point_dim = contour.shape[2]
        contour.resize((num_of_points, point_dim))


def main():
    parser = argparse.ArgumentParser(description="Find tiles in image.")
    parser.add_argument("-i", "--input", help="Image file path or GenTL endpoint", required=True)
    parser.add_argument("-o", "--output", help="Path to result file (JSON serialized COMPAS meshes)", required=True)
    parser.add_argument("-n", "--model_name", help="Name of the GenTL camera model to use")
    args = parser.parse_args()

    if args.input.endswith(".cti"):
        if not args.model_name:
            raise ValueError("When using a GenTL endpoint please provide a camera model name with '-n'.")
        capture_device = GenTlDevice(args.input, args.model_name)
    else:
        capture_device = SingleImageDevice(args.input)
    finder = TileFinder(capture_device, output_file=args.output)
    finder.run()
    capture_device.stop()


if __name__ == "__main__":
    main()
