import sys
import subprocess

def get_motifs(string, length, threshold):
    """Convert string to motifs using motif tool."""
    p = subprocess.Popen(
        ["./contiguous-fa-lang", str(length), str(threshold)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, _ = p.communicate(string + "\n")
    return out.strip().splitlines()

def greedy_select(input_file, output_file, seq_length, t, num_select):
    with open(input_file) as f:
        candidates = [line.strip() for line in f if line.strip()]

    motif_map = {}
    all_motifs = set()

    print("Building motif coverage map...")
    for i, line in enumerate(candidates):
        motifs = get_motifs(line, seq_length, t)
        motif_map[line] = set(motifs)
        all_motifs.update(motifs)
        if i % 100 == 0:
            print(f"Processed {i}/{len(candidates)} strings")

    selected = []
    used_motifs = set()

    print("Selecting top strings greedily...")
    for _ in range(num_select):
        best_line, best_gain = None, -1
        for line in candidates:
            if line in selected:
                continue
            gain = len(motif_map[line] - used_motifs)
            if gain > best_gain:
                best_gain = gain
                best_line = line
        if best_line:
            selected.append(best_line)
            used_motifs.update(motif_map[best_line])
            print(f"Selected {_ + 1}: Δmotifs={best_gain}")
        else:
            break

    with open(output_file, 'w') as out:
        for line in selected:
            out.write(line + '\n')

    print(f"Saved top {len(selected)} lines to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 greedy_selector.py <input.txt> <output.txt> <seq_len> <t> <num>")
        sys.exit(1)
    _, in_file, out_file, L, T, N = sys.argv
    greedy_select(in_file, out_file, int(L), int(T), int(N))
