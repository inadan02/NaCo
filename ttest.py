import numpy as np
from scipy.stats import ttest_ind

def load_scores(file_path):
    """Load integer scores from a text file."""
    with open(file_path) as f:
        return np.array([int(line.strip()) for line in f if line.strip().isdigit()])

# Load reactivity scores
random_scores = load_scores("/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_human.txt")
greedy_scores = load_scores("/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_greedy.txt")
genetic_scores = load_scores("/mnt/c/MASTER/SEM2/NaturalComputing/negative-selection-2020-master/model/reactivity_genetic.txt")

# Basic stats
print("=== Summary Statistics ===")
print(f"Random:  Mean = {random_scores.mean():.2f}, N = {len(random_scores)}")
print(f"Greedy:  Mean = {greedy_scores.mean():.2f}, N = {len(greedy_scores)}")
print(f"Genetic: Mean = {genetic_scores.mean():.2f}, N = {len(genetic_scores)}")

# T-tests
print("\n=== T-Tests (Welch's) ===")
t1, p1 = ttest_ind(greedy_scores, random_scores, equal_var=False)
print(f"Greedy vs Random:   t = {t1:.2f}, p = {p1:.3e}")

t2, p2 = ttest_ind(genetic_scores, random_scores, equal_var=False)
print(f"Genetic vs Random:  t = {t2:.2f}, p = {p2:.3e}")

t3, p3 = ttest_ind(genetic_scores, greedy_scores, equal_var=False)
print(f"Genetic vs Greedy:  t = {t3:.2f}, p = {p3:.3e}")
