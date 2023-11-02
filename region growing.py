from math import floor, sqrt, e, atan, degrees, isnan
from collections import defaultdict

import cv2
import numpy as np
"""region growing on image; user can use mouse click event to select target in image"""

INPUT_WINDOW_NAME = "Seed selection"
outputImage_WINDOW_NAME = "Region growing"
WHITE = 255
VARIANCE = 20  #threshold value for region growing


def regionGrowing(image, x, y):
    """Grows the region around seed pixel based on the variance to the seed"""
    (height, width) = image.shape # size of image
    outputImage = np.zeros((height, width))  # create outputImage image with 0-values
    outputImage[y, x] = WHITE # set seed in outputImage to be WHITE
    seed_value = image[y, x]  # get seed value from mouse event
    imageChanged = True  # until nothing has imageChanged in outputImage anymore
    while imageChanged:
        imageChanged = False
        for y in range(height):  # go through each pixel
            for x in range(width):
                if outputImage[y, x] == WHITE: # if pixel is part of region
                    neighbourPixels = [(x - 1, y),(x, y - 1), (x + 1, y), (x, y + 1),]
                    for (newX, newY) in neighbourPixels: # WHITE the 4 surrounding neighbor pixels
                        if 0 < newX < width and 0 < newY < height:  # check if neighbour pixel is in image
                            # and it has not been visisted and it's value is within specified variance
                            if outputImage[newY, newX] != WHITE and seed_value - VARIANCE < image[newY, newX] < seed_value + VARIANCE:
                                outputImage[newY, newX] = WHITE  # set outputImage to be WHITE
                                imageChanged = True # mark outputImage as imageChanged
    return outputImage


def applyRegionGrowing():
    image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
    def mouse_callback(event, x, y, flags, data):
        """Call for cv2 mouse click event"""
        if event == cv2.EVENT_LBUTTONDOWN:  # left mouse click event
            outputImage = regionGrowing(image, x, y) # grow region around clicked pixel
            cv2.imshow(outputImage_WINDOW_NAME, outputImage)  # display outputImage
            cv2.waitKey()
            cv2.destroyWindow(outputImage_WINDOW_NAME)
    cv2.namedWindow("Seed selection")  # show image
    cv2.setMouseCallback("Seed selection", mouse_callback)
    cv2.imshow("Seed selection", image)
    cv2.waitKey()

applyRegionGrowing()

