# FA26-COMP-671-Jasti

# Household Waste Classification Using CNNs

## Project Overview

This project investigates how reliably a Convolutional Neural Network (CNN) can classify real-world waste images into nine categories. It compares a baseline CNN with a CNN trained using data augmentation to understand whether augmentation improves robustness to changes in brightness and image rotation.

The project uses the RealWaste dataset and evaluates both models on the same test set.

## Research Question

How does a CNN's classification reliability change when classifying real-world waste images, and can data augmentation improve its robustness?

## Dataset

The project uses the RealWaste dataset, which contains 4,752 images across nine categories:

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

## Models

### 1. Baseline CNN

The baseline model uses three convolutional layers with 16, 32, and 64 filters, followed by a fully connected layer and a final classification layer.

### 2. CNN with Data Augmentation

The second model uses the same CNN architecture but applies random horizontal flips, rotations, and brightness and contrast adjustments to training images.

## Dataset Split

| Dataset | Images |
|---|---:|
| Training | 3,326 |
| Validation | 713 |
| Testing | 713 |
| **Total** | **4,752** |

The split uses random seed 42.

## Experimental Results

### Test Set Performance

| Model | Test Accuracy |
|---|---:|
| Baseline CNN | 67.18% |
| Augmented CNN | 60.17% |

The baseline CNN performed better on the original test images.

### Brightness Robustness

| Brightness Factor | Baseline CNN | Augmented CNN |
|---|---:|---:|
| 1.0 | 67.18% | 60.17% |
| 0.7 | 30.72% | 41.94% |
| 0.4 | 8.13% | 8.42% |

The augmented model performed better under moderate brightness reduction.

### Rotation Robustness

| Rotation | Baseline CNN | Augmented CNN |
|---|---:|---:|
| 0° | 67.18% | 60.17% |
| 15° | 34.64% | 58.63% |
| 30° | 32.54% | 53.30% |
| 45° | 27.35% | 48.95% |

The augmented model maintained higher accuracy at all tested nonzero rotation angles.

## Main Findings

- The baseline CNN achieved higher accuracy on the original test set.
- Data augmentation improved robustness to moderate brightness changes and image rotations.
- Both models struggled with very dark images.
- Miscellaneous Trash and Textile Trash were challenging categories for the augmented model.

These results show that higher accuracy on original images does not necessarily mean better robustness under changed image conditions.

## Project Structure

```text
household-waste-cnn/
├── figures/
│   ├── baseline_confusion_matrix.png
│   └── augmented_confusion_matrix.png
├── report/
│   └── final_report.md
├── results/
│   ├── baseline_history.json
│   ├── baseline_test_results.json
│   ├── augmented_history.json
│   ├── augmented_test_results.json
│   ├── brightness_robustness.json
│   └── rotation_robustness.json
├── src/
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── train_augmented.py
│   ├── evaluate_augmented.py
│   ├── evaluate_robustness.py
│   └── evaluate_rotation.py
├── README.md
└── PROJECT_PLAN.md
```

The project also uses a local dataset split file and locally saved model checkpoints. These files may be excluded from version control.

## Requirements

- Python 3.13 or a compatible Python version
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- Pillow

## Setup

Create and activate a virtual environment from the project directory.

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install torch torchvision matplotlib scikit-learn pillow
```

## Running the Project

Run scripts from the project root directory.

Inspect the dataset:

```powershell
python src/inspect_dataset.py
```

Prepare the dataset splits:

```powershell
python src/prepare_dataset.py
```

Train and evaluate the baseline model:

```powershell
python src/train_baseline.py
python src/evaluate_baseline.py
```

Train and evaluate the augmented model:

```powershell
python src/train_augmented.py
python src/evaluate_augmented.py
```

Run the robustness experiments:

```powershell
python src/evaluate_robustness.py
python src/evaluate_rotation.py
```

**Note:** The dataset must be available at the location expected by the scripts. Check the configuration and file paths before running them. Training scripts may take time to finish, especially on a CPU.

## Results and Figures

The `results/` directory stores experiment metrics and training history. The `figures/` directory contains the confusion matrices generated during evaluation.

The full project discussion, methodology, results, limitations, and references are available in [`report/final_report.md`](report/final_report.md).

## Limitations

The experiments use one train-validation-test split and a limited training period. The robustness evaluation covers only selected brightness factors and rotation angles. The results do not establish how the models will perform on every real-world waste image.

## Reference

Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. *Information, 14*(12), Article 633. https://doi.org/10.3390/info14120633
