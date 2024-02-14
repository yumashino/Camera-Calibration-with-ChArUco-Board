#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import math
import random
import datetime
from time import time

import yaml
import numpy as np
import cv2
import cv2.aruco as aruco
from tqdm import tqdm

random.seed(0)


def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)


def save_results_to_yaml(inputs, results, output_file_path):
    output_data = {'inputs': inputs, 'results': results}
    with open(output_file_path, 'w') as file:
        yaml.dump(output_data, file, sort_keys=False)


def generate_output_filename(input_path, suffix='results'):
    base, ext = os.path.splitext(input_path)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return f"{base}_{suffix}_{current_time}.yaml"


def calculate_camera_matrix(focal_length_mm, sensor_width_mm, sensor_height_mm, image_width_px, image_height_px):
    # Calculate the focal lengths in pixel units
    f_x = focal_length_mm * (image_width_px / sensor_width_mm)
    f_y = focal_length_mm * (image_height_px / sensor_height_mm)

    # Calculate the coordinates of the principal point (usually the image center)
    c_x = image_width_px / 2
    c_y = image_height_px / 2

    # Construct the camera matrix
    camera_matrix = np.asarray([[f_x, 0, c_x],
                                [0, f_y, c_y],
                                [0, 0, 1]], dtype=np.float32)

    return camera_matrix


def linear_interpolation(zoom_level, zoom_level_wide_end, zoom_level_tele_end, focal_length_wide_end_mm,
                         focal_length_tele_end_mm):
    # Estimate the focal length from the specified zoom level,
    # assuming that the focal length changes linearly between the telephoto and wide end

    zoom_level_ratio = (zoom_level - zoom_level_wide_end) / (zoom_level_tele_end - zoom_level_wide_end)

    interpolated_focal_length = focal_length_wide_end_mm + (
            zoom_level_ratio * (focal_length_tele_end_mm - focal_length_wide_end_mm))

    return interpolated_focal_length


def calculate_board_to_camera_distance_stats(tvecs):
    distances = [np.linalg.norm(tvec) for tvec in tvecs]
    mean_distance = np.mean(distances) / 1000  # mm -> m
    variance_distance = np.var(distances) / (1000 ** 2)  # mm^2 -> m^2
    std_deviation_distance = np.sqrt(variance_distance)  # m
    return mean_distance, variance_distance, std_deviation_distance


def main(config_path, input_video_path, max_frame_n, show_image=False, use_init_camera_matrix=False):
    config = load_config(config_path)

    # Calibration board parameters
    n_squares_x = config['board']['n_squares_x']
    n_squares_y = config['board']['n_squares_y']
    square_length = config['board']['square_length_mm']
    marker_length = config['board']['marker_length_mm']
    dictionary_name = config['board']['board_dictionary_name']
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    charuco_board = aruco.CharucoBoard([n_squares_x, n_squares_y], square_length, marker_length, dictionary)

    # Camera parameters
    camera_img_w = config['camera']['resolution_width_px']
    camera_img_h = config['camera']['resolution_height_px']
    sensor_width_mm = config['camera']['sensor_width_mm']
    sensor_height_mm = config['camera']['sensor_height_mm']
    image_width_px = camera_img_w
    image_height_px = camera_img_h

    # Lens parameters
    zoom_level = config['lens']['shooting_zoom_level']
    zoom_level_wide_end = config['lens']['zoom_level_wide_end']
    zoom_level_tele_end = config['lens']['zoom_level_tele_end']
    focal_length_wide_end_mm = config['lens']['focal_length_wide_end_mm']
    focal_length_tele_end_mm = config['lens']['focal_length_tele_end_mm']

    cap = cv2.VideoCapture(input_video_path)

    charuco_detector = aruco.CharucoDetector(charuco_board)

    print('Start detecting charuco corners...')
    obj_points_all = []
    img_points_all = []
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_id = -1
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_id += 1
        pbar.update(1)

        charucoCorners, charucoIds, markerCorners, markerIds = charuco_detector.detectBoard(img)

        if show_image:
            img2 = aruco.drawDetectedMarkers(img.copy(), markerCorners, markerIds)
            cv2.imshow('detected_markers', img2)

            img3 = aruco.drawDetectedCornersCharuco(img.copy(), charucoCorners, charucoIds, (255, 0, 0))
            cv2.imshow('detected_corners', img3)

            cv2.waitKey(0)

        if charucoIds is None:
            continue

        obj_point, imgPoint = charuco_board.matchImagePoints(charucoCorners, charucoIds)
        if obj_point.shape[0] > 8 and not None in obj_point:
            obj_points_all.append(obj_point)
            img_points_all.append(imgPoint)

    pbar.close()

    n_frames_found_corners = len(obj_points_all)
    print(f'Found charuco corners in {n_frames_found_corners} frames.\n')

    if 0 < max_frame_n < n_frames_found_corners:
        print(f'Use {max_frame_n} frames randomly.')

        # Create a list of random indices and get the corresponding elements from the original lists
        indices = list(range(n_frames_found_corners))
        random.shuffle(indices)
        indices = indices[:max_frame_n]

        obj_points_all = [obj_points_all[i] for i in indices]
        img_points_all = [img_points_all[i] for i in indices]

    if show_image:
        cv2.destroyAllWindows()

    print(f'Use initial camera matrix: {use_init_camera_matrix}')
    start = time()
    if use_init_camera_matrix:
        focal_length_at_zoom_level = linear_interpolation(zoom_level, zoom_level_wide_end, zoom_level_tele_end,
                                                          focal_length_wide_end_mm, focal_length_tele_end_mm)
        print(f'Estimated focal length at input zoom level \"{zoom_level}\": {focal_length_at_zoom_level:.2f} mm.')
        print('This focal length is used for initial camera matrix.')

        init_camera_matrix = calculate_camera_matrix(focal_length_at_zoom_level, sensor_width_mm, sensor_height_mm,
                                                     image_width_px, image_height_px)
        np.set_printoptions(precision=2, suppress=True)
        print(f'Initial camera matrix:\n{init_camera_matrix}')

        print('\nEstimating camera parameters. This may take a while...')
        init_dist_coeffs = None  # If specify, [K1, K2, P1, P2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_all, img_points_all,
                                                           (camera_img_w, camera_img_h),
                                                           init_camera_matrix, init_dist_coeffs,
                                                           flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    else:
        print('Estimating camera parameters. This may take a while...')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_all, img_points_all,
                                                           (camera_img_w, camera_img_h), None, None)
    end = time()
    print('Done! Elapsed time: {:.2f} s'.format(end - start))
    print(f'Re-projection RMS error: {ret:.4f}')

    fov_x = 2 * math.atan2(camera_img_w, 2 * mtx[0, 0]) * 180 / math.pi
    fov_y = 2 * math.atan2(camera_img_h, 2 * mtx[1, 1]) * 180 / math.pi
    shift_x = (mtx[0, 2] - camera_img_w / 2) / camera_img_w
    shift_y = (mtx[1, 2] - camera_img_h / 2) / camera_img_h

    # Distance from the calibration board to the camera
    mean_distance, variance_distance, std_deviation_distance = calculate_board_to_camera_distance_stats(tvecs)

    results = {
        'rms_re-projection_error': ret,
        'n_max_frames_found_corners': float(n_frames_found_corners),
        'distance_calibration-board_to_camera_mean_m': float(mean_distance),
        'distance_calibration-board_to_camera_variance_m': float(variance_distance),
        'distance_calibration-board_to_camera_stddev_m': float(std_deviation_distance),
        'camera_matrix': {
            'fx': float(mtx[0, 0]),
            'fy': float(mtx[1, 1]),
            'cx': float(mtx[0, 2]),
            'cy': float(mtx[1, 2])
        },
        'distortion': {
            'k1': float(dist[0][0]),
            'k2': float(dist[0][1]),
            'p1': float(dist[0][2]),
            'p2': float(dist[0][3]),
            'k3': float(dist[0][4])
        },
        'horizontal_fov': float(fov_x),
        'vertical_fov': float(fov_y),
        'horizontal_shift': float(shift_x),
        'vertical_shift': float(shift_y)
    }

    inputs = config  # Use the same inputs as the input config file
    inputs['program'] = {
        'input_config': config_path,
        'input_video': input_video_path,
        'n_frames_used_for_calibration': max_frame_n,
        'show_image': show_image,
        'use_init_camera_matrix': use_init_camera_matrix
    }

    output_yaml_path = generate_output_filename(config_path)
    save_results_to_yaml(inputs, results, output_yaml_path)
    print(f"Results saved: {output_yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Calibration with ChArUco Board')
    parser.add_argument('--config_path', '-c', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--input_video', '-i', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--max_frame_n', '-n', type=int, default=-1,
                        help='Max number of frames to use for calibration. If negative value, use all frames.')
    parser.add_argument('--show_image', '-s', action='store_true', help='Show detected markers and corners')
    parser.add_argument('--use_init_camera_matrix', '-u', action='store_true',
                        help='Use initial camera matrix for calibration')
    args = parser.parse_args()

    main(args.config_path, args.input_video, max_frame_n=args.max_frame_n,
         show_image=args.show_image, use_init_camera_matrix=args.use_init_camera_matrix)
