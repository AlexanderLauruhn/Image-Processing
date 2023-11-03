import cv2
import numpy as np
"""template matching by sum of squarred deviations"""

def ssd(reference, template):
    """Calculates sum of squared differences"""
    return np.sum((template - reference) ** 2)

image= cv2.imread("image.jpg") # load image
reference = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray
template = cv2.imread("section.jpg", cv2.IMREAD_GRAYSCALE) # load template image
(referenceHeight, referenceWidth) = reference.shape
(templateHight, templateWidth) = template.shape
heightDifference = referenceHeight - templateHight
widthDifference = referenceWidth - templateWidth
out_ssd = np.zeros((heightDifference, widthDifference)) # create arrays for error output
for y in range(heightDifference):
    for x in range(widthDifference):
        # calculate ssd and cor coefficient of image slice
        out_ssd[y, x] = ssd(reference[y:y+templateHight, x:x+templateWidth], template)
out_ssd = out_ssd / np.max(out_ssd)  # normalize ssd

(sy, sx) = np.unravel_index(np.argmin(out_ssd), out_ssd.shape)# get index of minimum error
resultImage = np.copy(image) # mark area using blue rectangle
cv2.rectangle(resultImage, (sx, sy), (sx + templateWidth, sy + templateHight), (255, 0, 0))
cv2.imshow("Template Matching", resultImage) # display image
cv2.waitKey()
