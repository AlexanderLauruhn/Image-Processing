import math

import cv2
import numpy as np

"""determine co-occurence matrix and calculate entropy, homogenity and contrast"""

ANGLES = np.deg2rad([0, 45, 90, 135])
DISTANCE = 1
VALUE_RANGE = 256

def cooccurrenceMatrix(image, angle):
    """calculate co-occurrence matrix of supplied image with angle angle and distance = 1"""
    (height, width) = image.shape
    # calculate offset of neighbor
    noff_x = math.ceil(math.cos(angle) * DISTANCE)
    noff_y = -math.ceil(math.sin(angle) * DISTANCE)
    cooccurrence = np.zeros((VALUE_RANGE, VALUE_RANGE)) # create output matrix
    for y in range(-noff_y, height):
        for x in range(0, width - noff_x):
            value = image[y, x] # get pixel and neighbor value
            neighbor = image[y + noff_y, x + noff_x]
            cooccurrence[value, neighbor] += 1  # increment co-occurrence value
    return cooccurrence / np.sum(cooccurrence)


def contrast(cooccurrence):
    """Calculate contrast texture measure"""
    (k, _) = cooccurrence.shape
    contrast = 0
    for i in range(k):
        for j in range(k):
            contrast += (i - j) ** 2 * cooccurrence[i, j]
    return contrast


def entropy(cooccurrence):
    """Calculate entropy texture measure"""
    (k, _) = cooccurrence.shape
    entropy = 0
    for i in range(k):
        for j in range(k):
            if cooccurrence[i, j] != 0:
                entropy -= cooccurrence[i, j] * math.log10(cooccurrence[i, j])
    return entropy

def homogeneity(cooccurrence):
    """Calculate homogeneity texture measure"""
    (k, _) = cooccurrence.shape
    homogeneity = 0
    for i in range(k):
        for j in range(k):
            homogeneity += cooccurrence[i, j] / (1 + math.fabs(i - j))
    return homogeneity



    # load images
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
for angle in ANGLES:
    cooccurrence = cooccurrenceMatrix(image, angle) # calculate co-occurrence matrix
     # print texture measures
    print("contrast:", contrast(cooccurrence))
    print("entropy:", entropy(cooccurrence))
    print("homogenity:", homogeneity(cooccurrence))
    print()

            # show coocurrence
    cv2.imshow("Co-occurence", cooccurrence / np.max(cooccurrence))
    cv2.waitKey()


