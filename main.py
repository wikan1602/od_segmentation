import cv2
import numpy as np

# Load the fundus image
fundus_image = cv2.imread('Images/training/drishtiGS_017.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(fundus_image, cv2.COLOR_BGR2GRAY)

# Normalize the image to the range 0-255
normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Apply thresholding
_, binary_image = cv2.threshold(normalized_image, 160, 255, cv2.THRESH_BINARY)

# Perform morphological closing
kernel = np.ones((100, 100), np.uint8)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the image to the bounding rectangle
optic_disc_cropped = fundus_image[y-30:y+h+30, x-30:x+w+30]

# Show the cropped image
cv2.imshow("Optic Disc Cropped", optic_disc_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()