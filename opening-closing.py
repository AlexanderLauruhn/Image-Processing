from math import floor
import cv2
import numpy as np
"""opening closing using erosion and dilatation to remove tiny stains from  black-white image"""
WHITE = 255
ITERATIONS = 2

def dilate(image, structure):
    """dilation of image using given structure"""
    (height, width) = image.shape #size of image
    (structureHeight, structureWidth) = structure.shape
    dilated = np.zeros(image.shape, np.uint8) # initialize output image
    sx_end = floor(structureWidth / 2)  # get start and end of structure
    sx_start = -sx_end
    sy_end = floor(structureHeight / 2)
    sy_start = -sy_end
    for y in range(height): # iterate all pixel
        for x in range(width):
            value = 0
            for sy in range(sy_start, sy_end + 1): # multiply section of image with structure and calculate sum
                for sx in range(sx_start, sx_end + 1):
                    if 0 <= y + sy < height and 0 <= x + sx < width:
                        value += img[y + sy, x + sx] * structure[sy - sy_start, sx - sx_start]
            if value > 0:
                dilated[y, x] = WHITE # dilate pixel
    return dilated


def erode(img, structure):
    """Erosion of supplied image using supplied structure"""
    (height, width) = img.shape
    (structureHeight, structureWidth) = structure.shape

    # initialize output
    eroded = np.zeros(img.shape, np.uint8)
    # claculate threshold
    high = np.sum(structure) * WHITE

    # get structure start and end
    sx_end = floor(structureWidth / 2)
    sx_start = -sx_end
    sy_end = floor(structureHeight / 2)
    sy_start = -sy_end

    # go through each pixel
    for y in range(height):
        for x in range(width):
            value = 0

            # multiply section of image with structure and calculate sum
            for sy in range(sy_start, sy_end + 1):
                for sx in range(sx_start, sx_end + 1):
                    if 0 <= y + sy < height and 0 <= x + sx < width:
                        value += img[y + sy, x + sx] * structure[sy - sy_start, sx - sx_start]

            # if value is threshold
            if value == high:
                # erode image
                eroded[y, x] = WHITE

    return eroded

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE) # load image
# initialize structure and iteration count
structure = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])


dilated = img # dilate image for i in ITERATIONS by structure
for _ in range(ITERATIONS):
    dilated = dilate(dilated, structure)
cv2.imshow("Dilatated", dilated)
cv2.waitKey()

eroded = dilated # erode image for number of ITERATIONS using structure
for i in range(ITERATIONS):
    eroded = erode(eroded, structure)
cv2.imshow("Eroded", eroded)
cv2.waitKey()

