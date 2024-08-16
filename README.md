# Image Segmentation and Clustering with KMeans

This project demonstrates how to perform image segmentation using KMeans clustering and how to visualize and save the segmented images. The code is implemented in Python using libraries such as OpenCV, Matplotlib, and scikit-learn.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Functionality](#functionality)
  - [Finding Optimal Clusters](#finding-optimal-clusters)
  - [Image Segmentation](#image-segmentation)
  - [Saving Images](#saving-images)
  - [Scatter Plot Visualization](#scatter-plot-visualization)
  - [Davies-Bouldin Index](#davies-bouldin-index)
- [License](#license)

## Overview

This repository contains code for segmenting an image using KMeans clustering. The process involves:
1. Loading an image and displaying it.
2. Reshaping the image data for clustering.
3. Determining the optimal number of clusters using the Elbow method.
4. Applying KMeans clustering for image segmentation.
5. Visualizing and saving the segmented image.
6. Comparing the original and segmented images using different compression formats.
7. Visualizing the color distribution of the original image and the cluster centroids.
8. Calculating the Davies-Bouldin Index to evaluate the clustering quality.

## Dependencies

To run this code, you need the following Python libraries:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `opencv-python`

You can install them using `pip`:

```bash
pip install numpy matplotlib scikit-learn opencv-python
