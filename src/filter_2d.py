"""
@file filter_2d.py
@brief Sample code that shows how to implement your own linear filters by using filter2D function
"""
import sys

import cv2 as cv
import numpy as np

# Global Variables
DELAY = 500
DDEPTH = -1
WINDOW_NAME = 'Filter 2D Demo'
DEFAULT_IMAGE = '../data/lena.png'


def main(argv: list[str]) -> None:
    # Get image name
    image_name = argv[0] if len(argv) > 0 else DEFAULT_IMAGE

    # Loads an image
    src_image = cv.imread(image_name, cv.IMREAD_COLOR)

    # Check if image is loaded
    if src_image is None:
        print('Error opening image!')
        print(f'Usage: filter_2d.py [{DEFAULT_IMAGE}] \n')
        return

    # Loop - Will filter the image with different kernel sizes each X miliseconds
    ind = 0
    while True:
        # Update kernel size for a normalized box filter with the range [3, 11]
        kernel_size = 3 + (2 * (ind % 5))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= (kernel_size * kernel_size)

        # Apply filter
        dst_image = cv.filter2D(src=src_image, ddepth=DDEPTH, kernel=kernel)
        cv.imshow(WINDOW_NAME, dst_image)

        c = cv.waitKey(DELAY)
        if c == 27:
            break

        ind += 1


if __name__ == "__main__":
    main(sys.argv[1:])
