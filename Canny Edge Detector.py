from math import floor, sqrt, e, atan, degrees, isnan
import cv2
import numpy as np
import warnings

"""Algorithm of Canny Edge Detection including smoothing, deviation, non-maxima suppression and hystersis"""

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

slice = 45
# the 8 directions
BOTTOM = 0
BOTTOMLEFT = 1
LEFT = 2
TOPLEFT = 3
TOP = 4
TOPRIGHT = 5
RIGHT = 6
BOTTOMRIGHT = 7

BLACK = 0
WHITE = 255


def gaussMask(sigma, n):
    """ create a gauss filter kernel with the size 2n+1 and sigma"""
    size = 2 * n + 1 # +1 to make size of kernel is odd
    halb = floor(size / 2) # half size
    # 1d coeffitient using standard distribution
    coeffitient = [e ** -((x - halb) ** 2 / (2 * sigma ** 2)) / sqrt(2 * sigma ** 2) for x in range(size)]
    factor = sum(coeffitient)  # normalize to sum 1
    normal = [x / factor for x in coeffitient]
    # return kernel with gauss filter
    return np.array([[x * y for x in normal] for y in normal])


def findEdges(image):
    """return value and direction based on sobel"""
    sobelMaskX = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]) # sobelfilter in both directions
    sobelMaskY = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    filterX = cv2.filter2D(image, -1, sobelMaskX) # apply Filter in horizontal direction
    filterY = cv2.filter2D(image, -1, sobelMaskY) # apply in vertical direction
    # intensity and direction
    intensity= np.array([[sqrt(x[0] ** 2 + y[0] ** 2) for (x, y) in zip(rx, ry)] for (rx, ry) in zip(filterX, filterY)])
    try:
        direction = np.array(
            [[atan(y[0] / x[0]) for (x, y) in zip(rx, ry)] for (rx, ry) in zip(filterX, filterY)])
    except:
        print("ERROR")
    return intensity, direction # return intensity and direction of edge pixel


def suppressNonMaxima(value, direction):
    """suppress non maxima of image"""
    (width, height) = value.shape # size
    # new array with 0-values in width*height size
    suppressedImage = np.zeros((width, height) , np.uint8)
    for y in range(height):
        for x in range(width):
            neighbourMax = 0
            # NaN direction = no gradient at this position
            if not isnan(direction[x, y]):
                # 8 directions
                slice_direction = floor((degrees(direction[x, y]) + (slice/2)) / slice)
                # values of neighbour pixel in given direction
                if slice_direction == BOTTOM or slice_direction == TOP:
                    if y > 0:
                        neighbourMax = max(neighbourMax, value[x, y - 1])
                    if y < height - 1:
                        neighbourMax = max(neighbourMax, value[x, y + 1])
                elif slice_direction == BOTTOMLEFT or slice_direction == TOPRIGHT:
                    if x > 0 and y > 0:
                        neighbourMax = max(neighbourMax, value[x - 1, y - 1])
                    if x < width - 1 and y < height - 1:
                        neighbourMax = max(neighbourMax, value[x + 1, y + 1])
                elif slice_direction == LEFT or slice_direction == RIGHT:
                    if x > 0:
                        neighbourMax = max(neighbourMax, value[x - 1, y])
                    if x < width - 1:
                        neighbourMax = max(neighbourMax, value[x + 1, y])
                elif slice_direction == TOPLEFT or slice_direction == BOTTOMLEFT:
                    if x < width - 1 and y > 0:
                        neighbourMax = max(neighbourMax, value[x + 1, y - 1])
                    if x > 0 and y < height - 1:
                        neighbourMax = max(neighbourMax, value[x - 1, y + 1])
            # save only values if no higher pixel values around
            if value[x, y] > neighbourMax:
                suppressedImage[x, y] = floor(value[x, y])
    return suppressedImage


def hysterese(value, lowerLimit, upperLimit): #upper and lower limit as parameter
    """apply hysteresis on image"""
    (width, height) = value.shape
    hysteresisValue = np.array(value, np.uint8)
    isValueChanged = True # boolean, if any value is changed
    while isValueChanged:
        # until no value is changed anymore
        isValueChanged = False
        #test all pixel in the double loop
        for y in range(height):
            for x in range(width):
                if hysteresisValue[x, y] > upperLimit:
                    # groesser als oberer grenzvalue
                    if hysteresisValue[x, y] < WHITE:
                        # if not, make white
                        hysteresisValue[x, y] = WHITE
                        isValueChanged = True # if not,leave loops
                    # check all neighbour pixel concerning to direction
                    if x > 0 and lowerLimit < hysteresisValue[x - 1, y] < upperLimit: #left
                        hysteresisValue[x - 1, y] = WHITE
                        isValueChanged = True
                    if x < width - 1 and lowerLimit < hysteresisValue[x + 1, y] < upperLimit: #right
                        hysteresisValue[x + 1, y] = WHITE
                        isValueChanged = True
                    if y > 0 and lowerLimit < hysteresisValue[x, y - 1] < upperLimit: #top
                        hysteresisValue[x, y - 1] = WHITE
                        isValueChanged = True
                    if y < height - 1 and lowerLimit < hysteresisValue[x, y + 1] < upperLimit: #bottom
                        hysteresisValue[x, y + 1] = WHITE
                        isValueChanged = True
                    if x > 0 and y > 0 and lowerLimit < hysteresisValue[x - 1, y - 1] < upperLimit: #topleft
                        hysteresisValue[x - 1, y - 1] = WHITE
                        isValueChanged = True
                    if x < width - 1 and y > 0 and lowerLimit < hysteresisValue[x + 1, y - 1] < upperLimit: #top right
                        hysteresisValue[x + 1, y - 1] = WHITE
                        isValueChanged = True
                    if x > 0 and y < height - 1 and lowerLimit < hysteresisValue[x - 1, y + 1] < upperLimit: #bottom left
                        hysteresisValue[x - 1, y + 1] = WHITE
                        isValueChanged = True
                    if x < width - 1 and y < height - 1 and lowerLimit < hysteresisValue[x + 1, y + 1] < upperLimit:#bottom right
                        hysteresisValue[x + 1, y + 1] = WHITE
                        isValueChanged = True
    # check all values between lower and upper limit
    for y in range(height):
        for x in range(width):
            if lowerLimit < hysteresisValue[x, y] < upperLimit:
                hysteresisValue[x, y] = BLACK
    return hysteresisValue # return the final image with canny edges


image = cv2.imread("image.jpg") # load image
gaussFiltered = cv2.filter2D(image, -1, gaussMask(0.5, 7)) #remove noise and smooth picture
(gradientValue, gradientDirection) = findEdges(gaussFiltered) # apply sobel filter
supressedValue = suppressNonMaxima(gradientValue, gradientDirection) # non maxima suppression
cv2.imshow("Hysterese", hysterese(supressedValue,50,50)) # show image after hysteresis
cv2.waitKey()

