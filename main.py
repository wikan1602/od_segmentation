import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fundus image
fundus_image = cv2.imread('Images/training/drishtiGS_094.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(fundus_image, cv2.COLOR_BGR2GRAY)

# Normalize the image to the range 0-255
normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Apply thresholding
_, binary_image = cv2.threshold(normalized_image, 160, 255, cv2.THRESH_BINARY)

# Perform morphological closing
kernel = np.ones((10, 10), np.uint8)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the image to the bounding rectangle
optic_disc_cropped = fundus_image[y-30:y+h+30, x-30:x+w+30]

# Create a subplot with 2 rows and 2 columns
plt.figure(figsize=(10, 8))

# Plot the original image
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(fundus_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Plot the gray image
plt.subplot(3, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')

# Plot the binary image
plt.subplot(3, 2, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')

# Plot the closed image
plt.subplot(3, 2, 4)
plt.imshow(closed_image, cmap='gray')
plt.title('Closed Image')

# Plot the closed image
plt.subplot(3, 2, 5)
plt.imshow(contours, cmap='gray')
plt.title('Contours Image')

# Plot the cropped image
plt.subplot(3, 2, 6)
plt.imshow(cv2.cvtColor(optic_disc_cropped, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')

plt.tight_layout()
plt.show()
