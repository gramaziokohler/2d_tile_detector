from abc import ABC
from abc import abstractmethod

import numpy
import cv2


class CaptureDevice(ABC):
    """An image source"""

    @abstractmethod
    def get_next_image(self):
        """Returns a single image from an image source"""
        pass

    @abstractmethod
    def stop(self):
        """Cleanup and free resources, if required"""
        pass


class SingleImageDevice(CaptureDevice):
    """Read an image from the given file path"""

    WIDTH_HEIGHT = (1024, 768)

    def __init__(self, filepath):
        self.image = cv2.imread(filepath)
        self.image = cv2.resize(self.image, self.WIDTH_HEIGHT)
        # self.image = cv.medianBlur(self.image, 5)

    def get_next_image(self) -> numpy.ndarray:
        """
        Returns a single image
        :return: image
        """
        return self.image

    def stop(self):
        # self.image.release() ?
        pass
