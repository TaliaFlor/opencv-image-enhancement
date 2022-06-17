import sys

import cv2 as cv
import numpy as np
from numpy import ndarray

# Global Variables
KERNEL_SIZE = 3
DDEPTH = cv.CV_16S
DEFAULT_IMAGE = '../data/lena.png'
WINDOW_NAME = 'Prewitt Demo - Simple Edge Detector'


def get_grad(src_image: ndarray, kernel_array: list[list[int]]) -> ndarray:
    kernel = np.array(kernel_array, dtype=int)
    return cv.filter2D(src=src_image, ddepth=DDEPTH, kernel=kernel)


def main(argv: list[str]) -> None:
    # Get image name
    image_name = argv[0] if len(argv) > 0 else DEFAULT_IMAGE

    # Load the image
    src_image = cv.imread(image_name, cv.IMREAD_COLOR)

    # Check if image is loaded
    if src_image is None:
        print('Error opening image!')
        print(f'Usage: prewitt_gradient.py [{image_name}] \n')
        return

    # Remove noise by blurring with a Gaussian filter
    src_image = cv.GaussianBlur(src_image, (KERNEL_SIZE, KERNEL_SIZE), 0)

    # Convert the image to grayscale
    gray_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    # Gradient-X
    kernel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    grad_x = get_grad(src_image=gray_image, kernel_array=kernel_x)

    # Gradient-Y
    kernel_y = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    grad_y = get_grad(src_image=gray_image, kernel_array=kernel_y)

    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    # Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow(WINDOW_NAME, grad)
    cv.waitKey(0)


if __name__ == "__main__":
    main(sys.argv[1:])
