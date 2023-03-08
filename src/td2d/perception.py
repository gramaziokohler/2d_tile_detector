from dataclasses import dataclass
from threading import Thread
from typing import Tuple

import cv2
from compas.geometry import Point, Vector

from .perception import GenTlDevice
from .calibration import OpenCVCalibrationData
from .calibration import OpenCVCalibrator
from .gui import UiManager


@dataclass
class Tile:
    centroid: Tuple[float, float, float]
    direction_vec: Vector


class TileLocator(Thread):
    """Locates a single tile in an image.

    1. Open a GenTL device
    2. Load previously created calibration data
    3. Start the run loop

    """

    def __init__(self, gentl_endpoint, camera_model, calibration_file):
        super().__init__()
        self._init_gui_values()
        self.ui_manager = UiManager("Display")
        self.device = GenTlDevice(camera_model, gentl_endpoint)
        self.calibrator = OpenCVCalibrator(OpenCVCalibrationData.from_file(calibration_file))
        self.calibrator.initialize(self.device.get_next_image())
        self.is_running = False
        self.current_tile = None

    @staticmethod
    def _init_gui_values():
        UiManager.THRESHOLD_SLIDER_MAX = 255
        UiManager.DEFAULT_THRESHOLD = 45  # best results: 45 for dark tiles, ~150 for bright ones

    def run(self):
        """While this loop is running, the program attempts to identify a tile in the image.

        1. Read image from input device
        2. Send image to get undistorted
        3. apply thresholding to image
        4. detect tile contour, centroid and corners using polygon approximation
        5. find tile orientation
        6. convert centroid and orientation vector to IRL.
        7. TODO: save to class member, this will run in the background and client code can take tile info when available

        """
        self.is_running = True
        self.ui_manager.start()
        while self.is_running:
            image = self.device.get_next_image()
            user_input = self.ui_manager.get_user_input()
            image = self.calibrator.undistortify(image)
            image = TileLocator.thresh_binary(image, user_input.threshold)

            polygon, centroid = self.approx_polygon(image)

            if polygon is not None:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(image, (polygon,), -1, (0, 255, 0), 3)
                cv2.circle(image, centroid, radius=10, color=(0, 0, 255), thickness=2)
                pixel_spcae_dir = self._calculate_dir_vec([(p[0, 0], p[0, 1]) for p in polygon])
                pixel_spcae_dir.scale(0.25)
                end_x = int(centroid[0] + pixel_spcae_dir.x)
                end_y = int(centroid[1] + pixel_spcae_dir.y)
                cv2.line(image, centroid, (end_x, end_y), (255, 0, 0), 3)
                irl_centroid = self.calibrator.pixel_to_irl_coords(centroid)
                irl_dir = Vector(*self.calibrator.pixel_to_irl_coords((pixel_spcae_dir.x, pixel_spcae_dir.y)))
                self.current_tile = Tile(irl_centroid, irl_dir)
                print(f"irl_centroid: {irl_centroid}")
            # cv2.circle(image, (1039, 617), radius=5, color=(0, 0, 255), thickness=2)
            self.ui_manager.show_image(image)
            if user_input.should_exit:
                self.stop()
        self.device.stop()

    @staticmethod
    def _calculate_dir_vec(points):
        """Get the 4 corners and use then to calculate the orientation vector of the tile.

        The orientation is pointing from the center towards the long dimension of the tile.
        Which way is random, but somewhat equivalent.

        Points are arranged clockwise.
        p0---------->p1
        .            |
        .            |
        p3. . . . . .p2
        1. create vector v1 from p0->p1
        2. create vector v2 from p1->p2
        3. which even of the two is longer, represents the orientation
        """
        p0 = Point(*points[0])
        p1 = Point(*points[1])
        p2 = Point(*points[2])
        v1 = Vector.from_start_end(p0, p1)
        v2 = Vector.from_start_end(p1, p2)
        if v1.length > v2.length:
            return v1
        return v2

    @staticmethod
    def approx_polygon(image, min_area=1000, max_area=100000, min_arc_length_factor=0.1):
        """Approximate the polygon of a square tile in the image.

        Only one contour -> one polygon with exactly 4 corners is expected to be found in the image.
        Anymore or less, this function shall return None.

        Use min and max area to filter the found contours by area. If no max area is specified, the whole image frame will be
        detected as one.

        min_arc_length_factor: the minimum curve length which shall be considered as a contour.
            This is calculated as a fraction of the p

        1. find contours
        2. filter with given min and max area (in pixel space)
        3. approximate the corners using and algorithm made by some dude whose initials are (probably were) DP
        4. get centroid using contour
        """
        contours, hierarchy = cv2.findContours(image, 1, 2)
        contours = tuple(cnt for cnt in contours if max_area > cv2.contourArea(cnt) > min_area)
        if len(contours) != 1:
            return None, None
        cnt = contours[0]
        epsilon = min_arc_length_factor * cv2.arcLength(cnt, True)
        polygon = cv2.approxPolyDP(cnt, epsilon, True)
        if len(polygon) != 4:
            return None, None
        centroid = TileLocator.get_centroid(cnt)
        return polygon, centroid

    @staticmethod
    def get_centroid(contour):
        m = cv2.moments(contour)
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    @staticmethod
    def _smoothen(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.medianBlur(image, 5)
        return image

    @staticmethod
    def thresh_to_zero(image, value):
        image = TileLocator._smoothen(image)
        _, image = cv2.threshold(image, value, 255, cv2.THRESH_TOZERO)
        return image

    @staticmethod
    def thresh_binary(image, value):
        image = TileLocator._smoothen(image)
        _, image = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
        return image

    def stop(self):
        self.is_running = False


def main():
    gentl_endpoint = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    camera_model = "Blackfly S BFS-PGE-19S4C"
    calibration_file = r"C:\Users\ckasirer\repos\upcycled_robotic_tiling\data\calibration\calibration20220908-163200.cal"  # you might need to give an absolute path here

    locator = TileLocator(gentl_endpoint, camera_model, calibration_file)
    locator.start()  # this doesn't block!

    # currently detected tile is available as
    # while True:
    #     tile = locator.current_tile
    #     if tile:
    #         print(f"current_tile: {tile}")


if __name__ == "__main__":
    main()
