#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2.aruco as aruco
from PIL import Image


def get_dict_name_as_str(ver):
    ver_str = None

    for name in dir(aruco):
        if getattr(aruco, name) == ver:
            ver_str = name
            break

    return ver_str


def main():
    dictionary_name = aruco.DICT_6X6_250
    dictionary = aruco.getPredefinedDictionary(dictionary_name)
    n_squares_x = 9
    n_squares_y = 5
    square_length_meter = 0.0233  # A4 210mm/9=23.3mm
    marker_length_meter = 0.0175  # 23.3*0.75=17.5mm
    charuco_board = aruco.CharucoBoard([n_squares_x, n_squares_y], square_length_meter, marker_length_meter, dictionary)

    # Generate ChArUco board
    board_img_w = 3508  # pixels (A4 210mm/25.4mm/inch x 300dpi)
    board_img_h = 2480  # pixels (A4 297mm/25.4mm/inch x 300dpi)
    board_img = charuco_board.generateImage((board_img_w, board_img_h))

    # Write baord image with DPI
    pil_img = Image.fromarray(board_img)
    pil_img.save(f'charuco_board_image_{n_squares_x}x{n_squares_y}_{get_dict_name_as_str(dictionary_name)}.png',
                 dpi=(300, 300))


if __name__ == '__main__':
    main()
