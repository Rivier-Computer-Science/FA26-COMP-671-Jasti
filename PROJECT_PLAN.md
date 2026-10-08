\# Deep Learning Final Project



\## Project Title



How Much Can a CNN Trust a Waste Image?

Evaluating CNN Reliability on Real-World Waste Images



\## Problem Statement



This project investigates the reliability of a Convolutional Neural Network

(CNN) for classifying real-world waste images.



Although a CNN may achieve strong performance on a standard test set, its

predictions may become less reliable when images contain variations in

lighting, orientation, composition, or other visual conditions.



This project will evaluate the classification performance of a CNN and

investigate how data augmentation affects its ability to generalize.



\## Motivation



Automated waste classification can support waste-sorting and recycling

processes by helping distinguish between different types of waste materials.



However, a model that performs well under limited or controlled image

conditions may not remain reliable when applied to real-world waste images.



Understanding the conditions under which the model succeeds or fails is

therefore an important part of evaluating a practical deep learning system.



\## Dataset



The project will use the RealWaste dataset.



The dataset details, class distribution, image characteristics, source,

license information, and train/validation/test split will be documented

after the dataset has been downloaded and inspected.



\## Research Question



How does a CNN's classification reliability change when classifying

real-world waste images, and can data augmentation improve its robustness?



\## Objectives



1\. Build a baseline CNN for waste image classification.

2\. Establish baseline performance on a held-out test set.

3\. Apply data augmentation and compare its performance with the baseline.

4\. Evaluate model performance under controlled visual variations.

5\. Identify conditions that cause performance degradation.

6\. Analyze incorrect predictions and identify common failure patterns.



\## Proposed Method



A Convolutional Neural Network (CNN) will be used as the primary deep

learning method because CNNs are appropriate for image classification tasks.



The project will begin with a simple CNN as the baseline model. A second

experiment will use the same CNN architecture with data augmentation.



Keeping the architecture consistent between these experiments will allow

the effect of data augmentation to be evaluated as a controlled comparison.



\## Baseline



The baseline will be a simple CNN trained using the standard preprocessing

pipeline without additional data augmentation.



The baseline will establish the initial classification performance against

which later experiments will be compared.



\## Controlled Comparison



The primary controlled comparison will be:



\- Baseline CNN without data augmentation

\- The same CNN with data augmentation



The purpose of this comparison is to determine whether data augmentation

improves generalization.



\## Robustness Evaluation



The trained model will be evaluated using controlled visual variations of

held-out test images.



Potential conditions may include:



\- Normal images

\- Reduced brightness

\- Increased brightness

\- Small rotations

\- Cropping

\- Other controlled visual variations supported by the dataset



The exact transformations will be finalized after examining the dataset.



\## Evaluation Metrics



The models will be evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion matrix



Performance across different experimental conditions will also be compared

to evaluate changes in model reliability.



\## Failure Analysis



Incorrect predictions will be examined to identify common failure patterns.



The analysis will investigate questions such as:



\- Which waste categories are most frequently confused?

\- Which visual conditions cause the largest performance decrease?

\- Are visually similar categories more difficult to classify?

\- What visual characteristics appear to contribute to incorrect predictions?



\## Expected Contribution



The project will provide an application-focused evaluation of CNN

classification performance rather than reporting only a single overall

accuracy.



The primary goal is to understand when the model performs reliably and when

its predictions become less reliable under different visual conditions.



\## Reproducibility



The project will document:



\- Dataset source

\- Dataset characteristics

\- Data preprocessing

\- Train/validation/test split

\- Model architecture

\- Training settings

\- Evaluation metrics

\- Experimental conditions

\- Dependencies

\- Instructions for reproducing the experiments



\## Project Status



Day 1: Project setup and dataset selection

