from math import floor, sqrt, e, atan, degrees, isnan
from collections import defaultdict
import cv2
import numpy as np

HISTOGRAM_LENGTH = 256
WHITE = 255

def histogram(image):
    """generate grayscale histogram of supplied image"""
    # initialize new histogram
    histogram = np.zeros((HISTOGRAM_LENGTH))
    for row in image: # iterate all pixel
        for pixel in row:
            histogram[pixel] += 1 # increment value of point in histogram
    return histogram


def sumsOfDividedHistogram(histogram, threshold):
    """Calulates the sums of the values from 0 to threshhold and from threshold to end"""
    return sum(histogram[:threshold]), sum(histogram[threshold+1:])


def mediansOfDividedHistogram(wheightHhistogram, threshold, n1, n2):
    """Calculate median values of the supplied weighted histogram left and right of the threshold"""
    # initialize medians
    medianBelowThreshold = 0
    medianAboveThreshold = 0
    # if there are no pixel on either side of threshold the specific median should be zero
    if n1 > 0:
        medianBelowThreshold = sum(wheightHhistogram[:threshold]) / n1
    if n2 > 0:
        medianAboveThreshold = sum(wheightHhistogram[1+threshold:]) / n2
    return medianBelowThreshold, medianAboveThreshold


def variance(factor, n1, n2, medianBelowThreshold, medianAboveThreshold):
    """Calculate variance of threshold using precalculated values"""
    return factor * n1 * n2 * (medianBelowThreshold - medianAboveThreshold) ** 2


def otsuapplyOtsuOnImage(image):
    """Determines optimal threshold of image using Otsu's method """
    # calculate histogram
    histogramArray = histogram(image)
    # calculate weighted histogram
    weightedHistogram = [i * h for (i, h) in enumerate(histogramArray)]
    factor = 1 / sum(histogramArray) ** 2  # calculate inverse of the sum of pixels squared as factor for calculation
    variances = np.zeros((len(histogramArray))) # initialize array with variances
    for value in range(len(histogramArray)): #iterate all grayscale values
        (n1, n2) = sumsOfDividedHistogram(histogramArray, value) # calculate sums
        (mu1, mu2) = mediansOfDividedHistogram(weightedHistogram, value, n1, n2)  # calculate medians
        variances[value] = variance(factor, n1, n2, mu1, mu2) # calculate variance using sums and medians and add to array
    return variances.argmax() # return index of max value

def applyOtsuOnImage(image, threshold):
    """Thresholds supplied image using supplied threshold value"""
    (height, width) = image.shape
    # initialize output image
    result = np.zeros((height, width))
    # go through each pixel in input image
    for y in range(height):
        for x in range(width):
            # if value in input image is below threshold, set output value to threshold high
            if image[y, x] < threshold:
                result[y, x] = WHITE
    return result


image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE) # read image in grayscale
otsuThreshold = otsuapplyOtsuOnImage(image) # calculate optimal threshold
output = applyOtsuOnImage(image, otsuThreshold) # threshold image
cv2.imshow("Otsu threshold", output) # display output
cv2.waitKey()
