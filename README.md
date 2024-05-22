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
- **Evaluation Metrics:** Accuracy, precision, recall, and F1-score were calculated for the best model.
- **Confusion Matrix:** Plotted to analyze classification errors and propose improvements.

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
   - Evaluate the classifier and plot a confusion matrix.
   - Propose improvements based on error analysis.

## Results

The project successfully identified and classified flower images with reasonable accuracy. The most important clusters were identified and ranked, contributing to improved classification performance.

## Future Work

- Implement more advanced clustering algorithms to further improve region extraction.
- Experiment with different classifiers and ensemble methods to enhance classification accuracy.
- Fine-tune the feature extraction process to capture more discriminative features.

## Contributors

- Soroush Fathi
- Soroush Pasandideh

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank the University of Oxford for providing the flower dataset and our professors for their guidance throughout this project.
