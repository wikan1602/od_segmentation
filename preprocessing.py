import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the fundus image
fundus_image = cv2.imread('Images/training/drishtiGS_088.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(fundus_image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram for grayscale image
hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Calculate the histogram for each channel (B, G, R)
hist_b = cv2.calcHist([fundus_image], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([fundus_image], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([fundus_image], [2], None, [256], [0, 256])

# Normalize the image to the range 0-255
normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(normalized_image)

# Calculate the histogram of the CLAHE image
clahe_hist = cv2.calcHist([clahe_image], [0], None, [256], [0, 256])

# Display the results
plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.imshow(fundus_image[..., ::-1])  # Convert BGR to RGB for display
plt.title('Original Fundus Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(normalized_image, cmap='gray')
plt.title('Normalized Grayscale Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.plot(hist_gray, color='k')
plt.title('Histogram (Grayscale)')

plt.subplot(2, 3, 5)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.plot(clahe_hist, color='k')
plt.title('Histogram CLAHE')

plt.tight_layout()
plt.show()
