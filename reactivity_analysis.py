import numpy as np
import matplotlib.pyplot as plt

def detailed_metrics(data, label):
    data = np.array(data)
    total = len(data)
    nonzero = data[data > 0]
    metrics = {
        "Set": label,
        "Total Strings": total,
        "Non-Zero Reactivity (%)": round(100 * len(nonzero) / total, 2),
        "Average Reactivity": round(nonzero.mean(), 2) if len(nonzero) > 0 else 0,
        "Max Reactivity": nonzero.max() if len(nonzero) > 0 else 0,
        "Median Reactivity": round(np.median(nonzero), 2) if len(nonzero) > 0 else 0,
        "Top 5% Threshold": round(np.percentile(data, 95), 2),
        "Top 1% Threshold": round(np.percentile(data, 99), 2),
    }
    return metrics

def run_metrics(input_file, label):
    with open(input_file) as f:
        data = [int(line.strip()) for line in f if line.strip().isdigit()]
    result = detailed_metrics(data, label)
    print(f"\nMetrics for {label}:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result

def plot_histogram(file_paths, labels, colors, output_file="reactivity_histogram2.png"):
    plt.figure(figsize=(10, 6))
    for path, label, color in zip(file_paths, labels, colors):
        with open(path) as f:
            data = [int(line.strip()) for line in f if line.strip().isdigit()]
        plt.hist(data, bins=50, alpha=0.5, label=label, color=color)
    plt.xlabel("Reactivity Score")
    plt.ylabel("Number of Sequences")
    plt.title("Reactivity Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

    print(f"\nHistogram saved as {output_file}")

# === RUNNING ===

# Metrics
run_metrics(
    "/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_greedy.txt",
    "Greedy Training"
)

run_metrics(
    "/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_human.txt",
    "Random Training"
)

run_metrics(
    "/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_genetic.txt",
    "Genetic Training"
)

# Histogram saved instead of shown
plot_histogram(
    [
        "/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_human.txt",
        "/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_greedy.txt",
        "/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_genetic.txt"
    ],
    ["Random", "Greedy", "Genetic"],
    ["blue", "green", "red"],
    output_file="reactivity_histogram2.png"
)

