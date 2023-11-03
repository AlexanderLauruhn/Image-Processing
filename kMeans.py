import math
import cv2
import numpy as np

"""Applies k-means algorithm to cluster image's color values"""
 
MAX_ITERATION = 8
BLACK = 0
WHITE = 255
K = 5


def kMeans(image, k):
    """Applies k-means clustering to supplied image with k-clusters"""
    height = image.shape[0]
    width = image.shape[1]
    channels = 1
    if len(image.shape) == 3:
        channels = image.shape[2]
    cluster = np.random.randint(BLACK, WHITE, (k, channels)) # generate random cluster
    r = None
    for i in range(MAX_ITERATION):
        r = np.zeros((height, width, k))
        # Step E
        for y in range(height):
            for x in range(width):
                d = np.sqrt(np.sum((image[y, x] - cluster) ** 2, axis=1))
                r[y, x, np.argmin(d)] = 1
        # Step M
        for j in range(k):
            sum = np.sum(r[:, :, j])
            if sum:
                if channels == 1:
                    cluster[j] = np.sum(r[:, :, j] * img[:, :]) / sum
                else:
                    for c in range(channels):
                        cluster[j, c] = np.sum(r[:, :, j] * image[:, :, c]) / sum
    # apply clustering to image
    clustering = np.zeros(image.shape, np.uint8)
    for y in range(height):
        for x in range(width):
            for j in range(k):
                if r[y, x, j] > 0:
                    clustering[y, x] = cluster[j]
    return clustering

image = cv2.imread("image.jpg")
cluster = kMeans(image, K) # apply clustering to image
cv2.imshow("Image", cluster)
cv2.waitKey()
cv2.destroyAllWindows()


