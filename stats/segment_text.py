# # segment_text.py
# import sys

# def segment_text(file_path, window=6):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         words = f.read().lower().split()
#     segments = [' '.join(words[i:i+window]) for i in range(len(words) - window + 1)]
#     return segments

# if __name__ == "__main__":
#     input_file = sys.argv[1]
#     output_file = sys.argv[2]
#     window_size = int(sys.argv[3])
#     segments = segment_text(input_file, window_size)
#     with open(output_file, 'w') as out:
#         for seg in segments:
#             out.write(seg + '\n')

# clean_segment_text.py


import sys
import re

def segment_text(path, out_path, window=6):
    with open(path, 'r') as f:
        raw = f.read().lower()
    raw = re.sub(r'[^a-z\s]', '', raw)  # remove punctuation
    words = raw.split()
    segments = [' '.join(words[i:i+window]) for i in range(len(words) - window + 1)]
    with open(out_path, 'w') as out:
        for seg in segments:
            out.write(seg + '\n')

if __name__ == "__main__":
    segment_text(sys.argv[1], sys.argv[2], int(sys.argv[3]))