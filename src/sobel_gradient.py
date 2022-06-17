"""
@file sobel_gradient.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import sys

import cv2 as cv
from numpy import ndarray

# Global Variables
WINDOW_NAME = 'Sobel Demo - Simple Edge Detector'
SCALE = 1
DELTA = 0
KERNEL_SIZE = 3
DDEPTH = cv.CV_16S
DEFAULT_IMAGE = '../data/lena.png'


def get_grad(src_image: ndarray, x_order: int, y_order: int) -> ndarray:
    return get_scharr_grad(src_image, x_order, y_order) if KERNEL_SIZE == 3 \
        else get_sobel_grad(src_image, x_order, y_order)


def get_sobel_grad(src_image: ndarray, x_order: int, y_order: int) -> ndarray:
    return cv.Sobel(src=src_image, ddepth=DDEPTH, dx=x_order, dy=y_order, ksize=KERNEL_SIZE, scale=SCALE, delta=DELTA)


def get_scharr_grad(src_image: ndarray, x_order: int, y_order: int) -> ndarray:
    return cv.Scharr(src=src_image, ddepth=DDEPTH, dx=x_order, dy=y_order)


def main(argv: list[str]) -> None:
    # Get image path
    image_name = argv[0] if len(argv) > 0 else DEFAULT_IMAGE

    # Load the image
    src_image = cv.imread(image_name, cv.IMREAD_COLOR)

    # Check if image is loaded
    if src_image is None:
        print('Error opening image!')
        print(f'Usage: sobel_gradient.py [{image_name}] \n')
        return

    # Remove noise by blurring with a Gaussian filter
    src_image = cv.GaussianBlur(src_image, (KERNEL_SIZE, KERNEL_SIZE), 0)

    # Convert the image to grayscale
    gray_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    # Gradient-X
    grad_x = get_grad(src_image=gray_image, x_order=1, y_order=0)

    # Gradient-Y
    grad_y = get_grad(src_image=gray_image, x_order=0, y_order=1)

    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    # Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow(WINDOW_NAME, grad)
    cv.waitKey(0)  # esc


if __name__ == "__main__":
    main(sys.argv[1:])
