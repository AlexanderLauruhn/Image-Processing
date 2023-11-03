import cv2
import numpy as np

KERNELSIZE = (3, 3)
SIGMA = 1.4
ALPHA = 0.06
THRESHOLD = 1000000
DISTANCE = 15
COLOR = (0, 0, 255) #red

def get_max_index(a):
    """Get indix of maximum in image"""
    return np.unravel_index(np.argmax(a), a.shape)

def calculate_structure_tensor(g):
    """Calculates structure tensor of grayscale images"""
    gx, gy = np.gradient(g)  # calculate gradient
    # return tensor as tuple
    return \
        (cv2.GaussianBlur(gx * gx, KERNELSIZE, SIGMA), cv2.GaussianBlur(gx * gy, KERNELSIZE, SIGMA)), \
        (cv2.GaussianBlur(gx * gy, KERNELSIZE, SIGMA),  cv2.GaussianBlur(gy * gy, KERNELSIZE, SIGMA))


def cornerResponse(structureTensor):
    """Calculates corner response function of structure tensor"""
    # extract tensor parts
    (gxgx, gygx), \
    (gxgy, gygy) = structureTensor
    determinate = gxgx * gygy - gxgy * gygx # calculate determinate and trace
    trace = gxgx + gygy
    return determinate - ALPHA * (trace * trace)  # return corner response function


def harris_corner_detector(img):
    """Detects and marks corners in image"""
    grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
    (height, width, _) = img.shape
    structureTensor = calculate_structure_tensor(grayscaleImage)   # calculate tensor
    cornerresponse = cornerResponse(structureTensor)  # calculate corner response function
    ret, threshold = cv2.threshold(cornerresponse, THRESHOLD, 0, cv2.THRESH_TOZERO) # apply thresholding
    y, x = get_max_index(threshold) # find maximum
    while threshold[y, x] > 0:  # while there are maxima in the threshold image
        # draw marker at maxima in output image
        cv2.drawMarker(img, (x, y), COLOR)
        # remove maxima from threshold image
        threshold[y, x] = 0
        # go through all neighboring pixels
        for sy in range(max(0, y - DISTANCE), min(y + DISTANCE, height)):
            for sx in range(max(0, x - DISTANCE), min(x + DISTANCE, width)):
                distance = (sx - x) + (sy - y)   # manhattan distance
                if distance <= DISTANCE: # if pixel is at most 10 pixels away
                    threshold[sy, sx] = 0 # remove pixel from threshold
        y, x = get_max_index(threshold) # get next maxima


image = cv2.imread("image.jpg") # load image
harris_corner_detector(image) # apply harris corner detection
cv2.imshow("Corner Detection", image)   # display image with markers
cv2.waitKey()

