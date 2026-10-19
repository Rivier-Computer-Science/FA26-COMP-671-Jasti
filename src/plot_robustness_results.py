import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def load_results(filename):
    with open(RESULTS / filename, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_comparison(title, x_values, baseline, augmented,
                    x_label, filename):
    plt.figure(figsize=(8, 5))

    plt.plot(x_values, baseline, marker="o", label="Baseline CNN")
    plt.plot(x_values, augmented, marker="o", label="Augmented CNN")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300)
    plt.close()

    print(f"Saved: figures/{filename}")


def main():
    brightness = load_results("brightness_robustness.json")
    rotation = load_results("rotation_robustness.json")

    # Convert fractional accuracies to percentages.
    brightness_levels = [1.0, 0.7, 0.4]
    baseline_brightness = [
        brightness["Baseline CNN"][str(level)] * 100
        for level in brightness_levels
    ]
    augmented_brightness = [
        brightness["Augmented CNN"][str(level)] * 100
        for level in brightness_levels
    ]

    plot_comparison(
        "CNN Robustness to Brightness Changes",
        brightness_levels,
        baseline_brightness,
        augmented_brightness,
        "Brightness Factor",
        "brightness_robustness_comparison.png",
    )

    rotation_angles = [0, 15, 30, 45]
    baseline_rotation = [
        rotation["Baseline CNN"][str(angle)] * 100
        for angle in rotation_angles
    ]
    augmented_rotation = [
        rotation["Augmented CNN"][str(angle)] * 100
        for angle in rotation_angles
    ]

    plot_comparison(
        "CNN Robustness to Image Rotation",
        rotation_angles,
        baseline_rotation,
        augmented_rotation,
        "Rotation Angle (degrees)",
        "rotation_robustness_comparison.png",
    )

    # Compare original test-set accuracy.
    models = ["Baseline CNN", "Augmented CNN"]
    accuracies = [
        brightness["Baseline CNN"]["1.0"] * 100,
        brightness["Augmented CNN"]["1.0"] * 100,
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, accuracies)
    plt.title("Original Test Set Accuracy")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.2f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(FIGURES / "model_accuracy_comparison.png", dpi=300)
    plt.close()

    print("Saved: figures/model_accuracy_comparison.png")
    print("All three comparison graphs were generated successfully.")


if __name__ == "__main__":
    main()
