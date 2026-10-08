\# Deep Learning Final Project



\## Project Title



How Much Can a CNN Trust a Waste Image?

Evaluating CNN Reliability Under Real-World Visual Distortions



\## Problem Statement



This project investigates the reliability of a Convolutional Neural Network (CNN)

for household waste image classification. Although a CNN may perform well on

standard test images, images encountered in practical settings can vary in

lighting, orientation, cropping, and visual background.



The project will examine how these realistic visual changes affect the

classification performance of a CNN and whether data augmentation can improve

its ability to generalize.



\## Motivation



Automated waste classification can support waste-sorting and recycling

processes. However, a model that performs well only on clean and consistent

images may not be reliable in practical situations.



Understanding how a CNN responds to changes in image conditions is therefore

important when evaluating whether a model can generalize beyond the conditions

seen during training.



\## Research Question



How does the reliability of a CNN-based household waste classifier change when

images are affected by realistic visual conditions such as lighting changes,

rotation, cropping, and background variation?



\## Objectives



1\. Build a baseline CNN for household waste classification.

2\. Establish baseline performance on a held-out test set.

3\. Apply data augmentation during training and compare it with the baseline.

4\. Evaluate model performance under controlled visual distortions.

5\. Identify which visual conditions cause the greatest performance degradation.

6\. Analyze incorrect predictions and identify common failure patterns.



\## Proposed Method



A Convolutional Neural Network (CNN) will be used as the primary deep learning

method because CNNs are appropriate for image classification tasks.



The project will begin with a simple CNN as the baseline model. A second

experiment will use the same CNN architecture with data augmentation during

training.



Keeping the architecture consistent between these experiments will allow the

effect of data augmentation to be evaluated as a controlled comparison.



\## Baseline



The baseline will be a simple CNN trained without data augmentation beyond the

basic preprocessing required for the images.



The baseline will establish the initial classification performance against

which later experiments will be compared.



\## Controlled Comparison



The primary controlled comparison will be:



\- Baseline CNN without data augmentation

\- The same CNN with data augmentation



The goal is to determine whether augmentation improves generalization.



\## Robustness Evaluation



The trained models will also be evaluated using controlled variations of

held-out test images.



Potential conditions include:



\- Normal images

\- Reduced brightness

\- Increased brightness

\- Small rotations

\- Cropped images

\- Background or visual clutter



The exact transformations will be finalized after examining the selected

dataset and available compute resources.



\## Evaluation Metrics



Model performance will be evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion matrix



Performance across the different visual conditions will also be compared to

identify changes in model reliability.



\## Failure Analysis



Incorrect predictions will be examined to identify common failure patterns.



The analysis will investigate questions such as:



\- Which waste categories are most frequently confused?

\- Which visual conditions cause the largest performance drop?

\- Are visually similar waste categories more difficult to classify?

\- What characteristics appear to contribute to incorrect predictions?



\## Expected Contribution



The project will provide an application-focused evaluation of CNN

classification performance rather than reporting only a single overall accuracy.



The main goal is to understand when the model performs reliably and when its

predictions become less reliable under realistic image variations.



\## Reproducibility



The project will document:



\- Dataset source

\- Data preprocessing

\- Train/validation/test split

\- Model architecture

\- Training settings

\- Evaluation metrics

\- Experimental conditions

\- Dependencies

\- Instructions for reproducing the experiments



\## Project Status



Day 1: Project setup and research design

