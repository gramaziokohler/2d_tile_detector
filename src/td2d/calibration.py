from dataclasses import dataclass
from typing import Any, Tuple
import cv2
import numpy


@dataclass
class OpenCVCalibrationData:
    """Loads calibration co-efficients from calibration output from the file generated by the calibration script.

    Camera matrix: contains the intrinsic parameters of the camera, as found during calibration.
    dist_coefficients: values which are used to counteract the distortion caused by the lens.
    rotation_vecs: the rotation part of the extrinsic matrix
    translate_vecs: the translation part of the extrinsic matrix
    new_camera_matrix: a new optimized camera matrix, with compensation for lens distortion
    region_of_interest: subset of the image which should be now distortion free. x_start and y_start
    in the original image along with new width and height

    inv_camera_matrix: invert of the camera matrix used during the translation of pixel to world coordinates
    scaling factor: a big mystery, but used as well for the translation of pixel to world coordinates

    """

    camera_matrix: Any
    dist_coefficients: Any
    rotate_vecs: Any
    translate_vecs: Any
    new_camera_matrix: Any
    region_of_interest: Any
    inv_camera_matrix: Any
    inv_new_camera_matrix: Any

    inv_rotate_mat = None
    scaling_factor = None

    @classmethod
    def from_file(cls, path):
        """Loads camera matrix and distortion coefficients."""
        # FILE_STORAGE_READ
        import os

        assert os.path.exists(path)
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        camera_matrix = cv_file.getNode("K").mat()
        dist_matrix = cv_file.getNode("D").mat()
        rot_vector = cv_file.getNode("R").mat()
        trans_vector = cv_file.getNode("T").mat()
        new_camera_matrix = cv_file.getNode("NK").mat()
        roi = cv_file.getNode("ROI").mat()
        roi = (roi[0, 0], roi[1, 0], roi[2, 0], roi[3, 0])
        inv_camera_matrix = numpy.linalg.inv(camera_matrix)  # maybe this should be
        inv_new_camera_matrix = numpy.linalg.inv(new_camera_matrix)
        cv_file.release()
        return OpenCVCalibrationData(
            camera_matrix,
            dist_matrix,
            rot_vector,
            trans_vector,
            new_camera_matrix,
            roi,
            inv_camera_matrix,
            inv_new_camera_matrix,
        )

    def calculate_scaling_factor(self):
        # https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
        # convert rotation vector to rotation matrix (3x3)
        # stack translation vector to rotation matrix, result => extrinsic matrix (3x4)
        # dot multiply newcamera matrix by extrinsic matrix (3x4) to get projection matrix (3x4)
        # dot perspective matrix with origin [X,Y,Z,1] get => [X', Y', Z'] vector
        # Z' is the scaling factor
        rot_matrix = cv2.Rodrigues(self.rotate_vecs)[0]
        self.inv_rotate_mat = numpy.linalg.inv(rot_matrix)
        extrinsic_mat = numpy.column_stack((rot_matrix, self.translate_vecs))
        projection_mat = self.new_camera_matrix.dot(extrinsic_mat)
        # 7.5, 4.1, 55.5 explained:
        # 55.5 cm is the distance from the camera's lens to the physical point on the plane
        # where the cameras cx and cy point is projected.
        # 7.5, 4.1 are the x and y distance in cm from a rather arbitrarily chosen origin
        # on the plane to the above mentioned point.
        xyz1 = numpy.array([[7.5, 4.1, 55.5, 1.0]], dtype=numpy.float32).reshape((4, 1))
        result_xyz = projection_mat.dot(xyz1)
        self.scaling_factor = result_xyz[2, 0]


class OpenCVCalibrator:
    def __init__(self, calibration_data: OpenCVCalibrationData, image_size: Tuple[int, int] = None):
        self._calibration_data = calibration_data
        self.image_width = int(calibration_data.region_of_interest[2])
        self.image_height = int(calibration_data.region_of_interest[3])
        self.roi_x = int(calibration_data.region_of_interest[0])
        self.roi_y = int(calibration_data.region_of_interest[1])
        self.is_initialized = False

    def initialize(self, image: numpy.ndarray):
        self._calibration_data.calculate_scaling_factor()
        _ = self.undistortify(image)
        self.is_initialized = True
        print("### Calibration init info ###")
        print(f"Cropped ROI size: w:{self.image_width} h:{self.image_height}")
        print(f"Scaling factor: {self._calibration_data.scaling_factor}")
        print(f"Camera matrix: {self._calibration_data.camera_matrix}")
        print(f"New camera matrix: {self._calibration_data.new_camera_matrix}")
        print("### Calibration init info ###")

    def undistortify(self, image: numpy.ndarray):
        """Remove distortion from image using the loaded distortion coefficients.

        Returns the distortion free portion of the original image, cropped using the region of interest parameters.
        """
        result = cv2.undistort(
            image,
            self._calibration_data.camera_matrix,
            self._calibration_data.dist_coefficients,
            None,
            self._calibration_data.new_camera_matrix,
        )

        # crop the image
        return result[
            self.roi_y : self.roi_y + self.image_height, self.roi_x : self.roi_x + self.image_width  # noqa: E203
        ]

    def pixel_to_irl_coords(self, pixel_coords: Tuple[int, int]) -> Tuple[float, float, float]:
        """The coordinates received are in camera space, but x,y origin is not necessarily at the center of the camera.

        The plan, position a tile until its centroid reads closest possible to 0, 0.
        Then measure the location in REAL real world coordinates using the robot.
        """
        # https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
        # dot of pixel coords with inverse camera mtx
        # dot of result with inverse of translation vecs
        # dot of result with inverse of rotation
        uv1 = numpy.array([[pixel_coords[0], pixel_coords[1], 1]], dtype=numpy.float32).reshape(3, 1)
        scaled_uv1 = self._calibration_data.scaling_factor * uv1
        irl_xyz = self._calibration_data.inv_new_camera_matrix.dot(
            scaled_uv1
        )  # should this be the inverse new cam matrix instead?
        irl_xyz = irl_xyz - self._calibration_data.translate_vecs
        irl_xyz = self._calibration_data.inv_rotate_mat.dot(irl_xyz)
        return (
            irl_xyz[0, 0],
            irl_xyz[1, 0],
            irl_xyz[2, 0],
        )
