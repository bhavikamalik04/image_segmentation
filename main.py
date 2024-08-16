import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
import cv2
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import FunctionTransformer

# Load and display the image
image_path = "path_to_your_image.jpg"  # Replace with the path to your image
image = mpimg.imread(image_path)
plt.imshow(image)
plt.axis('off')  # Hide the axis
plt.show()

# Reshape the image for clustering
x = image.reshape(-1, 3)

# Function to determine the optimal number of clusters
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()

# Find the optimal number of clusters
find_optimal_clusters(x, 10)

# Define a function to transform data to approximate Manhattan distance
def manhattan_transform(X):
    return np.sqrt(np.abs(X))

# Apply the transformation to your data
transformer = FunctionTransformer(manhattan_transform)
x_transformed = transformer.transform(x)

# Apply KMeans with the chosen number of clusters
kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
kmeans.fit(x_transformed)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

# Display the segmented image
plt.imshow(segmented_img / 255)
plt.axis('off')  # Hide the axis
plt.show()

# Save the original image with compression
cv2.imwrite("original_image.jpeg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# Convert segmented image to uint8 and save with different compression settings
segmented_img_uint8 = (segmented_img / 255 * 255).astype(np.uint8)

# Save with a lower JPEG quality to test compression
cv2.imwrite("segmented_image_low_quality.jpeg", cv2.cvtColor(segmented_img_uint8, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 50])

# Save as PNG for comparison
cv2.imwrite("segmented_image.png", cv2.cvtColor(segmented_img_uint8, cv2.COLOR_RGB2BGR))

# Check the file sizes
original_size = os.path.getsize("original_image.jpeg")
segmented_size_jpeg = os.path.getsize("segmented_image_low_quality.jpeg")
segmented_size_png = os.path.getsize("segmented_image.png")

print(f"Original Image Size: {original_size} bytes")
print(f"Segmented Image Size (Low Quality JPEG): {segmented_size_jpeg} bytes")
print(f"Segmented Image Size (PNG): {segmented_size_png} bytes")

# Display scatter plot of original and centroid colors
original_rgb_values = x
centroid_rgb_values = kmeans.cluster_centers_
plt.figure(figsize=(8, 8))
plt.scatter(original_rgb_values[:, 0], original_rgb_values[:, 1], c=original_rgb_values / 255, s=1, marker='o')
plt.scatter(centroid_rgb_values[:, 0], centroid_rgb_values[:, 1], c=centroid_rgb_values / 255, s=200, marker='x')
plt.title('Scatter Plot of Original Image Colors and Cluster Centroids')
plt.xlabel('Red')
plt.ylabel('Green')
plt.show()

# Print original and centroid RGB values
print("Original RGB Values (first 10 values):")
print(original_rgb_values[:10])
print("Centroid RGB Values:")
print(centroid_rgb_values)

# Calculate Davies-Bouldin Index
dbi = davies_bouldin_score(x, kmeans.labels_)
print(f"Davies-Bouldin Index: {dbi:.2f}")
