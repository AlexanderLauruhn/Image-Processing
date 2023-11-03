import cv2
import numpy as np

REGION = 15
FULL_ROTATION = 360
SAMPLES = 100
THRESHOLD = 0.3
COLOR = (255, 0, 255)

def hough_circles(edges, minimumRadius, maximumRadius):
    """Apply hough transformation for circles"""
    (height, width) = edges.shape
    # initialize accumulator
    accumulator = np.zeros((maximumRadius - minimumRadius, height + 2 * maximumRadius, width + 2 * maximumRadius))
    for r in range(minimumRadius, maximumRadius):  # for each possible radius
        angles = np.radians(np.arange(0, FULL_ROTATION, FULL_ROTATION / SAMPLES))   # compute each angle
        # compute x and y coordinates for each angle
        circle = np.column_stack((r * np.sin(angles),
                                  r * np.cos(angles))).astype(np.int8)
        rl, rh = r - maximumRadius + 1, r + maximumRadius + 1
        for y, x in np.argwhere(edges):
            for [cy, cx] in circle: # point in circle
                accumulator[r - minimumRadius, y - cy, x - cx] += 1  # increment accumulator at center of suspected circle
    detectedCircles = list()
    r, y, x = np.unravel_index(np.argmax(accumulator), accumulator.shape)  # find maximum index
    while accumulator[r, y, x] > THRESHOLD * SAMPLES:
        radius = r + minimumRadius
        detectedCircles.append((radius, y, x))
        accumulator[:, y-radius:y+radius, x-radius:x+radius] = 0      # remove surrounding values
        r,y,x =  np.unravel_index(np.argmax(accumulator), accumulator.shape)
    return detectedCircles


img = cv2.imread("image.jpg") # load image
gauss = cv2.GaussianBlur(img, (3, 3), 2) # apply gauss filter
edges = cv2.Canny(gauss, 100, 200) # apply canny edge detector
circles = hough_circles(edges, 20, 40) # find circles
sum = 0
for (radius, y, x) in circles:
    cv2.circle(img, (x, y), radius, COLOR)  # draw circle
cv2.imshow("Output", img) # display image
cv2.waitKey()
