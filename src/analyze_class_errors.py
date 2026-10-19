
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

def main():
    with (RESULTS / "baseline_misclassified.json").open(
        encoding="utf-8"
    ) as file:
        mistakes = json.load(file)

    # Count mistakes by actual category.
    counts = Counter(item["true"] for item in mistakes)
    categories = sorted(counts, key=counts.get, reverse=True)

    print("Misclassified images by actual category:")
    for category in categories:
        print(f"{category}: {counts[category]}")

    # Plot number of errors for each category.
    plt.figure(figsize=(10, 6))
    plt.bar(categories, [counts[c] for c in categories])
    plt.title("Baseline CNN: Misclassified Images by True Category")
    plt.xlabel("Actual Waste Category")
    plt.ylabel("Number of Incorrect Predictions")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output = FIGURES / "baseline_errors_by_category.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"\nSaved: {output.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
