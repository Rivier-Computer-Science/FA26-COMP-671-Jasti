# How Much Can a CNN Trust a Waste Image?

## Evaluating CNN Reliability on Real-World Waste Images

## 1. Project Overview

Waste classification is an important application of computer vision that can support waste sorting and recycling. However, real-world waste images can be challenging because of variations in lighting, orientation, background, and visual similarity between waste categories.

This project evaluates the reliability of a Convolutional Neural Network (CNN) trained to classify real-world waste images. It compares a baseline CNN with a CNN trained using data augmentation to investigate whether augmentation improves robustness under image transformations.

**Research Question:** How does a CNN's classification reliability change when classifying real-world waste images, and can data augmentation improve its robustness?

## 2. Dataset

This project uses the **RealWaste dataset**, introduced by Single et al. (2023), which contains real-world images of waste materials.

- **Total images:** 4,752
- **Number of classes:** 9
- **Task:** Multiclass image classification

### Waste Categories

1. Cardboard
2. Food Organics
3. Glass
4. Metal
5. Miscellaneous Trash
6. Paper
7. Plastic
8. Textile Trash
9. Vegetation

The dataset presents a realistic classification challenge because some waste categories share similar visual characteristics.

**Dataset reference:** Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*. Information, 14(12), Article 633. https://doi.org/10.3390/info14120633

## 3. Methodology

### 3.1 Data Preparation

The dataset was divided into training, validation, and test sets using a fixed random seed of 42.

| Dataset split | Number of images | Percentage |
|---|---:|---:|
| Training | 3,326 | 70% |
| Validation | 713 | 15% |
| Testing | 713 | 15% |
| **Total** | **4,752** | **100%** |

The same data split was used to compare both models. The test set was used for final evaluation and robustness experiments.

### 3.2 Baseline CNN

The baseline CNN was trained using the original training images with the preprocessing defined in the training pipeline. Its performance provides a reference point for evaluating the effect of data augmentation.

### 3.3 CNN with Data Augmentation

A second CNN was trained using data augmentation to expose the model to variations in training images. The purpose was to investigate whether augmentation could improve generalization and robustness to changes in image appearance and orientation.

### 3.4 Robustness Evaluation

Both models were evaluated on the same test images under the following conditions:

- **Original images:** Standard test-set evaluation.
- **Brightness changes:** Brightness factors of 1.0, 0.7, and 0.4.
- **Rotation changes:** Rotations of 0°, 15°, 30°, and 45°.

Accuracy was used to compare model performance under each condition.

## 4. Experimental Results

### 4.1 Original Test-Set Performance

| Metric | Baseline CNN | Augmented CNN |
|---|---:|---:|
| Test accuracy | 67.18% | 60.17% |
| Macro precision | 0.69 | 0.62 |
| Macro recall | 0.65 | 0.58 |
| Macro F1-score | 0.66 | 0.58 |
| Weighted F1-score | 0.67 | 0.59 |

The baseline CNN achieved higher accuracy on the original test set. Therefore, data augmentation did not improve performance on unmodified images in this experiment.

![Original test accuracy comparison](figures/model_accuracy_comparison.png)

**Figure 1. Original Test Accuracy Comparison.** The baseline CNN achieved 67.18% accuracy, compared with 60.17% for the augmented CNN.

### 4.2 Brightness Robustness

| Brightness factor | Baseline CNN | Augmented CNN |
|---|---:|---:|
| 1.0 (original brightness) | 67.18% | 60.17% |
| 0.7 (reduced brightness) | 30.72% | 41.94% |
| 0.4 (strongly reduced brightness) | 8.13% | 8.42% |

The augmented CNN performed better at a brightness factor of 0.7, achieving an improvement of approximately 11.22 percentage points over the baseline CNN.

At a brightness factor of 0.4, both models performed poorly. This suggests that severe brightness reduction remains a significant challenge for both models.

![Brightness robustness comparison](figures/brightness_robustness_comparison.png)

**Figure 2. Brightness Robustness Comparison.** The graph compares both models across three brightness conditions.

### 4.3 Rotation Robustness

| Rotation angle | Baseline CNN | Augmented CNN |
|---|---:|---:|
| 0° | 67.18% | 60.17% |
| 15° | 34.64% | 58.63% |
| 30° | 32.54% | 53.30% |
| 45° | 27.35% | 48.95% |

The augmented CNN performed substantially better under all tested nonzero rotation conditions. At 45°, it achieved 48.95% accuracy, compared with 27.35% for the baseline CNN—a difference of 21.60 percentage points.

These results suggest that the augmentation strategy improved robustness to image rotation, even though the augmented model had lower accuracy on the original test set.

![Rotation robustness comparison](figures/rotation_robustness_comparison.png)

**Figure 3. Rotation Robustness Comparison.** The augmented CNN retained higher accuracy as the rotation angle increased.

## 5. Failure Analysis

The baseline CNN correctly classified 479 of the 713 test images and misclassified 234 images.

### 5.1 Misclassified Examples

Examining incorrect predictions helps identify cases where the model struggles to distinguish between visually similar waste categories.

![Baseline CNN misclassified examples](figures/baseline_misclassified_examples.png)

**Figure 4. Baseline CNN Misclassified Examples.** Examples of test images for which the baseline CNN predicted the wrong category.

### 5.2 Errors by Category

The number of baseline CNN errors varied across the nine waste categories.

| Actual category | Number of misclassified images |
|---|---:|
| Plastic | 40 |
| Metal | 34 |
| Miscellaneous Trash | 33 |
| Textile Trash | 33 |
| Cardboard | 30 |
| Glass | 21 |
| Food Organics | 18 |
| Paper | 18 |
| Vegetation | 7 |

Plastic had the largest number of errors in absolute terms, followed by Metal. Vegetation had the fewest errors. However, the categories contain different numbers of test images, so error counts alone do not indicate which category had the lowest classification accuracy.

![Baseline CNN errors by category](figures/baseline_errors_by_category.png)

**Figure 5. Baseline CNN Errors by Actual Category.** The graph compares the number of incorrect predictions across the nine waste categories.

## 6. Discussion

The experiments demonstrate that classification accuracy on original images does not necessarily reflect robustness under changed image conditions.

The baseline CNN achieved better accuracy on the original test set, but its accuracy decreased substantially when brightness or rotation transformations were applied.

The augmented CNN showed better performance under moderate brightness reduction and all tested nonzero rotation angles. However, both models struggled with strong brightness reduction.

These findings suggest that data augmentation can improve robustness to certain image transformations, but its effectiveness depends on the transformation and the training configuration. A model that performs better on standard test images is not necessarily the most robust model under real-world variations.

## 7. Limitations

This project has several limitations:

- The evaluation uses one dataset and one fixed train-validation-test split.
- The experiments focus on classification accuracy rather than a broader range of robustness metrics.
- Only selected brightness factors and rotation angles were tested.
- Severe brightness reduction caused poor performance in both models.
- The results do not establish robustness to every real-world condition, such as blur, occlusion, camera differences, or complex backgrounds.
- The observed differences apply to the models and experimental settings used in this project; additional runs would be needed to assess variability across random seeds.

## 8. Conclusion

This project evaluated a baseline CNN and an augmented CNN on the RealWaste dataset.

The baseline CNN achieved **67.18% test accuracy**, while the augmented CNN achieved **60.17%** on the original test images. Despite its lower original accuracy, the augmented CNN performed better under moderate brightness reduction and image rotation.

The largest observed rotation-robustness difference occurred at 45°, where the augmented CNN achieved 48.95% accuracy compared with 27.35% for the baseline CNN.

Overall, the results indicate that data augmentation can improve robustness to specific image transformations, but it does not guarantee better performance across every evaluation condition.

## 9. Project Structure

```text
household-waste-cnn/
├── data/
│   └── splits.json
├── figures/
│   ├── baseline_confusion_matrix.png
│   ├── augmented_confusion_matrix.png
│   ├── model_accuracy_comparison.png
│   ├── brightness_robustness_comparison.png
│   ├── rotation_robustness_comparison.png
│   ├── baseline_misclassified_examples.png
│   └── baseline_errors_by_category.png
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
│   ├── evaluate_rotation.py
│   ├── plot_robustness_results.py
│   ├── analyze_misclassified.py
│   └── analyze_class_errors.py
├── .gitignore
└── README.md
```

The dataset, local split configuration, and model checkpoint files should remain local when excluded by `.gitignore`, unless there is a specific reason to share them.

## 10. How to Run

### Requirements

- Python 3
- PyTorch
- torchvision
- Matplotlib
- scikit-learn
- Pillow

Install the required packages in your Python environment:

```bash
pip install torch torchvision matplotlib scikit-learn pillow
```

### Run the experiments

Run the following commands from the project root, after configuring the dataset path in the scripts if necessary.

```bash
python src/inspect_dataset.py
python src/prepare_dataset.py
python src/train_baseline.py
python src/evaluate_baseline.py
python src/train_augmented.py
python src/evaluate_augmented.py
python src/evaluate_robustness.py
python src/evaluate_rotation.py
python src/plot_robustness_results.py
python src/analyze_misclassified.py
python src/analyze_class_errors.py
```

Training scripts create model checkpoints and training-history files. Evaluation and analysis scripts generate metrics, robustness results, and figures.

**Note:** The exact dataset path and training settings must match those configured in the source scripts.

## 11. References

1. Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. *Information, 14*(12), Article 633. https://doi.org/10.3390/info14120633
2. Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G