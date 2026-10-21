# FA26-COMP-671-Jasti

# Evaluating CNN Reliability on Real-World Waste Images

**Course:** COMP-671, Fall 2026  
**Author:** Sravya Jasti

## Project Overview

This project investigates the reliability of convolutional neural networks (CNNs) for classifying real-world waste images into nine categories. It compares a baseline CNN with a CNN trained using data augmentation and evaluates how brightness changes and image rotations affect classification accuracy.

## Research Question

How does a CNN's classification reliability change when classifying real-world waste images, and can data augmentation improve its robustness?

## Dataset

This project uses the **RealWaste** dataset, which contains 4,752 images across nine categories:

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

Dataset source: [UCI Machine Learning Repository — RealWaste](https://doi.org/10.24432/C5SS4G)

The dataset was split into training (3,326 images), validation (713 images), and testing (713 images), using random seed 42. Images were resized to 128 × 128 pixels.

## Models and Experiments

### Baseline CNN

The baseline model uses three convolutional layers with 16, 32, and 64 filters, followed by a fully connected layer. It was trained using cross-entropy loss and the Adam optimizer, with a learning rate of 0.001, batch size of 32, and up to 10 epochs.

### CNN with Data Augmentation

The second model uses the same architecture and training settings, with random horizontal flips, rotations up to 10 degrees, and brightness and contrast changes during training.

### Robustness Evaluation

Both models were evaluated using:
- Brightness factors of 1.0, 0.7, and 0.4.
- Rotation angles of 0°, 15°, 30°, and 45°.
- Confusion matrices and category-level classification errors.

## Main Results

| Evaluation | Baseline CNN | Augmented CNN |
|---|---:|---:|
| Original test accuracy | 67.18% | 60.17% |
| Accuracy at brightness 0.7 | 30.72% | 41.94% |
| Accuracy at brightness 0.4 | 8.13% | 8.42% |
| Accuracy at rotation 45° | 27.35% | 48.95% |

The baseline achieved higher accuracy on the original test images. The augmented model performed better under moderate brightness reduction and the tested image rotations. Both models struggled with severely darkened images.

## Repository Contents

- `src/` — source code and experiment scripts.
- `figures/` — generated figures and plots.
- `results/` — evaluation results.
- `report/` — project report files.
- `requirements.txt` — Python dependencies.

## Setup and Reproducibility

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Download the RealWaste dataset from the source linked above.
5. Follow the dataset preparation and training instructions in the project scripts.

Run scripts from the project root. Check each script's arguments and dataset-path settings before running it; exact commands may depend on your local dataset location.

## Limitations

The experiments used one dataset split, one random seed, and a limited number of training epochs. The robustness tests covered only selected brightness levels and rotation angles. Results may differ on other datasets and real-world sorting environments.

## Report

The final written report is available in the `report/` directory.
