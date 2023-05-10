"""
https://github.com/gramaziokohler/clamp_controller/blob/master/src/visual_docking/calibration.py
"""
import time
import argparse
from typing import Tuple
from typing import Any

import cv2
import numpy as np
from compas_urt.perception import GenTlDevice, SingleImageDevice

from aruco_markers import save_coefficients


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
window_name = 'VideoStream'


class VideoCaptureAdapter:
    """
    A wrapping interface for GenTlDevice to make it compatible (to some extent) with OpenCV's VideoCapture.
    """
    def __init__(self, gentl_device: GenTlDevice):
        self.device = gentl_device

    def read(self) -> Tuple[Any, np.ndarray]:
        return None, self.device.get_next_image()

    def release(self) -> None:
        self.device.stop()


def calibrate(url, square_size_cm, width=9, height=6, camera_model_name=None):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size_cm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    image_size = None

    if url.endswith(".cti"):
        if not camera_model_name:
            raise ValueError("Please provide the camera's model name when working with a GenTL endpoint!")
        vcap = VideoCaptureAdapter(GenTlDevice(gentl_endpoint=url, model_name=camera_model_name))
        # vcap = VideoCaptureAdapter(SingleImageDevice(r"c:\Users\ckasirer\Downloads\left22.jpg"))
    else:
        vcap = cv2.VideoCapture(url)

    try:
        is_running = True
        while is_running:

            _ret, frame = vcap.read()
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                is_running = False

            if key & 0xFF == ord('c'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

                # If found, add object points, image points (after refining them)
                if ret:
                    print('Image captured, object points found')
                    objpoints.append(objp)

                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)

                    # # Draw and display the corners
                    cv2.drawChessboardCorners(frame, (width, height), corners2, ret)
                    cv2.imshow(window_name, frame)
                    image_size = frame.shape[:2]
                    cv2.waitKey(1000)

                    print("objpoints:\n")
                    for p in objpoints:
                        print(p)

                    print("\nimgpoints:\n")
                    for p in imgpoints:
                        print(p)
                else:
                    print('Image captured but NO object points found')

    finally:
        if vcap:
            vcap.release()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    h, w = image_size
    newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    cv2.destroyAllWindows()

    return ret, mtx, dist, rvecs, tvecs, newcam_mtx, roi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--url', type=str, required=True, help='url of the camera stream')
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')
    parser.add_argument("--camera_model_name", "-m", type=str, required=False, help="Name of the camera model (e.g. Blackfly S BFS-PGE-19S4C)")

    args = parser.parse_args()
    ret, mtx, dist, rvecs, tvecs, new_cam_mat, roi = calibrate(args.url, args.square_size, args.width, args.height, args.camera_model_name)
    result_filepath, extension = args.save_file.split(".")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_filepath = ".".join([result_filepath + timestamp, extension])
    save_coefficients(mtx, dist, rvecs[0], tvecs[0], new_cam_mat, roi, result_filepath)
    print("Calibration is finished. RMS: ", ret)
