import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fundus image
fundus_image = cv2.imread('Images/training/drishtiGS_068.png')

# Check if the image is loaded properly
if fundus_image is None:
    raise ValueError("Image not found or path is incorrect")

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

# Check if any contours were found
if not contours:
    raise ValueError("No contours found in the image")

# Get the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Ensure the crop area is within the image bounds
y1, y2 = max(0, y-30), min(fundus_image.shape[0], y+h+30)
x1, x2 = max(0, x-30), min(fundus_image.shape[1], x+w+30)

# Crop the image to the bounding rectangle
optic_disc_cropped = fundus_image[y1:y2, x1:x2]

# Create a subplot with 3 rows and 2 columns
plt.figure(figsize=(12, 10))

# Plot the original image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(fundus_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Plot the gray image
plt.subplot(2, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')

# Plot the binary image
plt.subplot(2, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')

# Plot the closed image
plt.subplot(2, 3, 4)
plt.imshow(closed_image, cmap='gray')
plt.title('Closed Image')

# Draw the contours on the original image
contour_image = fundus_image.copy()
cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

# Plot the contours image
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Contours Image')

# Plot the cropped image
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(optic_disc_cropped, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')

plt.tight_layout()
plt.show()
