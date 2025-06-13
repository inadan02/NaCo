import random
import os

def split_file(source_path, train_path, val_path,
               val_fraction = 0.20, seed = 123):

    # read and keep only non-empty lines
    with open(source_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    rnd = random.Random(seed)
    rnd.shuffle(lines)

    n_val = int(len(lines) * val_fraction)
    val_lines   = lines[:n_val]
    train_lines = lines[n_val:]

    # ensure destination folders exist
    os.makedirs(os.path.dirname(train_path), exist_ok = True)
    os.makedirs(os.path.dirname(val_path),   exist_ok = True)

    with open(train_path, "w") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(val_path, "w") as f:
        f.write("\n".join(val_lines) + "\n")

# create four files in one go
split_file("/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/test_llm_clean.txt",
           "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/train_llm_clean.txt",
           "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/val_llm_clean.txt")

split_file("/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/train_human_clean.txt",
           "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/train_human_clean_train.txt",
           "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/val_human_clean.txt")
