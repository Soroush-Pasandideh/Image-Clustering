# Flower Classification Project

## Overview

This project aims to classify images of flowers from the Oxford 102 Flowers Dataset using clustering and classification techniques. The project is divided into two main phases: feature extraction using clustering and classification based on extracted features.

## Dataset

The dataset used in this project is the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/). It contains images of flowers from 102 different classes.

## Phase 1: Feature Extraction and Clustering

### Objectives
   
1. **Extract Valuable Image Regions:**
   - Extract color and spatial features from each pixel.
   - Perform clustering to identify regions containing flowers.
   - Use K-means clustering algorithm with various parameters to optimize region extraction.

2. **Evaluate Clustering:**
   - Ensure extracted regions have spatially close pixels and similar colors.
   - Avoid extracting regions that are too small or too large.
   - Adjust the importance of spatial and color features to optimize clustering.

### Methods

- **K-means Clustering:** Applied to extract significant image regions.
- **Parameter Tuning:** Various algorithms and parameters were tested to find the best clustering configuration.
- **Evaluation Metrics:** Both qualitative and quantitative metrics (at least two) were used to evaluate the clustering performance.

## Phase 2: Feature Extraction from Regions and Classification

### Objectives

1. **Feature Extraction:**
   - Extract color statistical features and shape features from each region.
   - Construct feature vectors for images by clustering all features and creating histograms of features per image.

2. **Classification:**
   - Use the extracted feature vectors to train a classifier.
   - Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.

### Methods

- **Statistical and Shape Features:** Extracted from identified regions.
- **Feature Vector Construction:** Histograms of clustered features were created for each image.
- **Classifier Training:** Trained a classification algorithm to predict flower classes based on feature vectors.

## Implementation

### Notable Functions

- **`calculate_mean_imgs_clstrs_features`**: Computes the mean feature vector for clusters.
- **`calculate_mean_imgs_clstrs_except_index`**: Computes the mean feature vector excluding a specific cluster.
- **`cal_classify_results`**: Classifies images and outputs classification results.
- **`check_clusters_importance`**: Determines the importance of clusters by evaluating classification confidence changes when clusters are removed.
- **`remove_least_important_clusters`**: Removes the least important clusters from the feature set.

### Steps

1. **Feature Extraction:**
   - Extract features for each region and create histograms.
   - Use clustering to identify and rank the importance of each region.

2. **Classification:**
   - Train a classifier with the constructed feature vectors.
   - Propose improvements based on error analysis.

## Results

The project successfully identified and classified flower images with notable efficiency. The key achievements include:

### Clustering and Feature Extraction
1. **Optimal Clustering Configuration**: Through extensive testing and parameter tuning of the K-means clustering algorithm, we identified the best configuration that effectively extracted meaningful regions from the flower images.
2. **Feature Extraction**: Extracted both color statistical and shape features from identified regions, which were then used to construct comprehensive feature vectors for each image.

### Importance of Clusters
1. **Cluster Analysis**: Using custom functions such as `check_clusters_importance` and `remove_least_important_clusters`, we evaluated the importance of each cluster by measuring changes in classification confidence when clusters were removed.
2. **Confidence Impact**: We observed that removing the least important clusters led to a minimal decrease in classification confidence, which helped refine the feature set to focus on the most critical regions.
<img src="./images/2" width="300" height="100">
<img src="./images/3" width="300" height="100">
<img src="./images/4" width="300" height="100">
<img src="./images/5" width="300" height="100">
### Additional Results
1. **Reduction in Feature Set**: By removing the least important clusters, we reduced the feature set size by 20% without significantly impacting classification performance, resulting in a more efficient model.
2. **Visualization**: Important clusters for each image were visualized, providing insights into which regions the classifier focused on for making decisions.
