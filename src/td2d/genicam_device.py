from numpy import ndarray
from harvesters.core import Harvester

from .device import CaptureDevice


class GenTlDeviceError(Exception):
    pass


class GenTlDevice(CaptureDevice):
    """
    Get live stream frames from a GenICam device via a GenTL producer.
    Input color format is expected to be RBG8.

    Input gets resized to WIDTH_HEIGHT
    """

    WIDTH_HEIGHT = (800, 600)

    def __init__(self, model_name: str, gentl_endpoint: str):
        self.endpoint = gentl_endpoint
        self.model_name = model_name
        self.harvester = Harvester()
        self.device = None
        self.buffer = None
        self._start_capture_device()

    def _start_capture_device(self):
        self.harvester.add_file(self.endpoint)
        self.harvester.update()

        if len(self.harvester.device_info_list) == 0:
            raise GenTlDeviceError("No devices detected!")
        try:
            self.device = self.harvester.create({"model": self.model_name})
        except ValueError:
            raise GenTlDeviceError(f"No device with name: {self.model_name} is present.")
        self.device.start(run_as_thread=True)

    def get_next_image(self) -> ndarray:
        """Get the next available frame from the camera stream"""
        if self.buffer:
            self.buffer.queue()  # data in buffer will not be available anymore after this

        self.buffer = self.device.fetch()
        image = self.buffer.payload.components[0]
        data = image.data
        data = data.reshape(image.height, image.width, int(image.num_components_per_pixel))
        return data

    def stop(self) -> None:
        """Cleanup"""
        if self.buffer:
            self.buffer.queue()
        if self.device:
            self.device.destroy()
        if self.harvester:
            self.harvester.reset()
