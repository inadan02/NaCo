import collections
import matplotlib.pyplot as plt

# Load the optimized training set from the genetic or greedy output
genetic_file = "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/my_ga_results/run29/best_training_set.txt"  # Replace with your actual path
llm_corpus_file = "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/test_llm_clean.txt"  # Full LLM dataset
# Load lines
with open(genetic_file) as f:
    genetic_lines = [line.strip() for line in f if line.strip()]

with open(llm_corpus_file) as f:
    llm_lines = [line.strip() for line in f if line.strip()]

# Count motif frequencies in full LLM dataset
llm_freq = collections.Counter(llm_lines)

# Count motif occurrences in the optimized set
genetic_freq = collections.Counter(genetic_lines)

# Get frequency of each selected motif in full LLM corpus
motif_frequencies = [(motif, llm_freq[motif]) for motif in genetic_lines]

# Sort by frequency
motif_frequencies.sort(key=lambda x: x[1])

# Prepare for plotting
top_motifs = motif_frequencies[:30]  # 30 lowest frequency motifs

motifs, freqs = zip(*top_motifs)

# Plot
plt.figure(figsize=(12, 6))
plt.barh(motifs, freqs, color='skyblue')
plt.xlabel("Frequency in LLM Corpus")
plt.title("Least Frequent Motifs in Genetic Training Set")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.show()


from collections import Counter

# Load motifs
with open("motifs_genetic.txt") as f:
    motifs = [line.strip() for line in f if line.strip()]

# Count frequencies
motif_counts = Counter(motifs)

# Show top 20
print("Top 20 most frequent motifs:")
for motif, count in motif_counts.most_common(20):
    print(f"{motif}: {count}")
