import sys

import cv2 as cv

# Global Variables
DEFAULT_IMAGE = '../data/lena.png'
WINDOW_NAME = 'Prewitt Demo - Simple Edge Detector'


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


if __name__ == "__main__":
    main(sys.argv[1:])
