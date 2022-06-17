"""
@file filter_2d.py
@brief Sample code that shows how to implement your own linear filters by using filter2D function
"""
import sys

import cv2 as cv
import numpy as np


def main(argv):
    window_name = 'Filter 2D Demo'

    default_image = '../data/lena.png'
    image_name = argv[0] if len(argv) > 0 else default_image

    # Loads an image
    src = cv.imread(image_name, cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(f'Usage: filter_2d.py [{default_image}] \n')
        return -1

    # Initialize ddepth argument for the filter
    ddepth = -1

    # Loop - Will filter the image with different kernel sizes each 0.5 seconds
    ind = 0
    while True:
        # Update kernel size for a normalized box filter
        kernel_size = 3 + (2 * (ind % 5))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= (kernel_size * kernel_size)

        # Apply filter
        dst = cv.filter2D(src, ddepth, kernel)
        cv.imshow(window_name, dst)

        c = cv.waitKey(500)
        if c == 27:
            break

        ind += 1

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
