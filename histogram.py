import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the fundus image
fundus_image = cv2.imread('Images/training/drishtiGS_017.png')

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

# Calculate the histogram for normalized grayscale image
hist_norm = cv2.calcHist([normalized_image], [0], None, [256], [0, 256])

# Plot the histograms
plt.figure(figsize=(12, 6))

# Plot grayscale histogram
plt.subplot(1, 3, 1)
plt.plot(hist_gray, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Plot RGB histograms
plt.subplot(1, 3, 2)
plt.plot(hist_b, color='blue', label='Blue Channel')
plt.plot(hist_g, color='green', label='Green Channel')
plt.plot(hist_r, color='red', label='Red Channel')
plt.title('RGB Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

# Plot normalized grayscale histogram
plt.subplot(1, 3, 3)
plt.plot(hist_norm, color='black')
plt.title('Normalized Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()
