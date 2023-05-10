"""
Sources:
https://gist.github.com/naoki-mizuno/c80e909be82434ddae202ff52ea1f80a
https://github.com/opencv/opencv_contrib/blob/e247b680a6bd396f110274b6c214406a93171350/modules/aruco/samples/calibrate_camera_charuco.cpp

"""
import time
import os
from copy import deepcopy
from argparse import ArgumentParser

import cv2 as cv
import numpy as np

from compas_urt.perception import GenTlDevice


CAMERA = "Blackfly S BFS-PGE-31S4C"
GENTL_ENDPOINT = "C:\\Program Files\\MATRIX VISION\\mvIMPACT Acquire\\bin\\x64\\mvGenTLProducer.cti"
BOARD_DIMS = (12, 9)
SQUARE_SIZE_M = 0.03
MARKER_SIZE_M = 0.023
DICT_SIZE = cv.aruco.DICT_5X5_250
WINDOW = "Charuco Calibration"
RESULT_DIR = "C:\\Users\\ckasirer\\repos\\compas_urt\\data\\calibration\\"


def calibrate():
    device = GenTlDevice(CAMERA, GENTL_ENDPOINT)
    board, dictionary = create_board()
    detector = cv.aruco.ArucoDetector(dictionary)

    corners_list = []
    ids_list = []
    frame_list = []

    while True:
        image = device.get_next_image()
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_size = grayscale.shape
        corners, ids, rejected = detector.detectMarkers(grayscale)
        image_copy = deepcopy(image)
        if ids is not None and len(ids) > 0:
            corners, ids, rejected, recovered = cv.aruco.refineDetectedMarkers(grayscale, board, corners, ids, rejected)
            ret, c_corners, c_ids = cv.aruco.interpolateCornersCharuco(corners, ids, grayscale, board)
            cv.aruco.drawDetectedMarkers(image_copy, corners)
            cv.aruco.drawDetectedCornersCharuco(image_copy, c_corners, c_ids)


        cv.imshow(WINDOW, image_copy)

        key = cv.waitKey(20)
        if key == 27:
            break
        elif key == ord('c'):
            # record frame and its detected markers
            if len(ids) == 0:
                print("No ids detected!")
                continue
            print("adding frame to list")
            corners_list.append(c_corners)
            ids_list.append(c_ids)
            frame_list.append(grayscale)

    # start calibration using collected data
    if not (corners_list or ids_list or frame_list):
        print("no calibration info captured")

    corners_list = [x for x in corners_list if len(x) >= 4]
    ids_list = [x for x in ids_list if len(x) >= 4]
    ret, camera_matrix, dist_coeff, rvec, tvec = cv.aruco.calibrateCameraCharuco(
        corners_list, ids_list, board, image_size, None, None
    )

    print(f"ret:{ret}, camera_matrix:{camera_matrix}, dist_coeff:{dist_coeff}, rvec:{rvec}, tvec:{tvec}")

    save_coefficients(camera_matrix, dist_coeff, rvec, tvec, RESULT_DIR)

def create_board():
    d = cv.aruco.getPredefinedDictionary(DICT_SIZE)
    return cv.aruco.CharucoBoard(BOARD_DIMS, SQUARE_SIZE_M, MARKER_SIZE_M, d), d


def save_coefficients(mtx, dist, r_vecs, t_vecs, dir_path):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(dir_path, f"calib_result_{timestr}.dat")
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.write("R", r_vecs[0])
    cv_file.write("T", t_vecs[0])
    cv_file.release()


if __name__ == '__main__':
    calibrate()
