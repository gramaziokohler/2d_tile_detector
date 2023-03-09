from dataclasses import dataclass
from copy import deepcopy
from typing import Tuple

import numpy
import cv2


@dataclass
class UserInput:
    """
    Stores user input state relevant for the tile finding operation
    """

    min_area = 0
    max_area = 0
    threshold = 0
    should_exit = False
    should_save = False

    def reset_booleans(self) -> None:
        """
        Reset user input boolean flags so that they are not persistent
        """
        self.should_save = False
        self.should_exit = False


class UiManager:
    """
    Wraps the windowing and user input functionality of openCV
    """

    THRESHOLD_SLIDER_MAX = 10
    DEFAULT_THRESHOLD = 5
    TILE_AREA_SLIDER_MAX = 60000
    DEF_MAX_TILE_AREA = TILE_AREA_SLIDER_MAX // 2
    DEF_MIN_TILE_AREA = 100
    WAIT_FOR_INPUT_MS = 10

    def __init__(self, window_name="Window"):
        self.window_name = window_name
        self.user_input = UserInput()

    def get_user_input(self) -> UserInput:
        """
        Wait for user in put WAIT_FOR_USER
        :return:
        """
        self.user_input.reset_booleans()
        k = cv2.waitKey(self.WAIT_FOR_INPUT_MS)
        if k == ord("q") or self._is_window_closed():
            self.user_input.should_exit = True
        if k == ord("s"):
            self.user_input.should_save = True

        return self.user_input

    def start(self):
        """
        Start an OpenCV window and setup control callbacks
        :return:
        """
        cv2.namedWindow(self.window_name)
        self._setup_controls()

    def _is_window_closed(self):
        try:
            return cv2.getWindowProperty(self.window_name, 0) == -1
        except cv2.error:
            return True

    def _setup_controls(self):
        # GUI event handlers
        def on_change(value):
            self.user_input.threshold = value

        def min_on_change(value):
            self.user_input.min_area = value

        def max_on_change(value):
            self.user_input.max_area = value

        cv2.createTrackbar("threshold", self.window_name, self.DEFAULT_THRESHOLD, self.THRESHOLD_SLIDER_MAX, on_change)

    def draw_selected_tiles(self, on_img: numpy.ndarray, tiles: Tuple[numpy.ndarray]) -> None:
        """
        Draw the found contours on top of the given image
        :param on_img: image to draw on
        :param tiles: the contours to draw
        """
        img_copy = deepcopy(on_img)
        cv2.drawContours(img_copy, tiles, -1, (0, 255, 0), 3)
        self.show_image(img_copy)

    def show_image(self, image):
        cv2.imshow(self.window_name, image)
