# 2020-negative-selection
[![DOI](https://zenodo.org/badge/140730506.svg)](https://zenodo.org/badge/latestdoi/140730506)

This repository contains the simulation code for the manuscript "Is T Cell Negative Selection a Learning Algorithm?".
It includes the original implementation of a negative selection model based on finite automata, as well as a new extension that builds on this baseline to address a real-world classification task.

In the original work, negative selection was applied to peptides and synthetic strings to study self–nonself discrimination in the immune system. This extended version applies the same core principles to a natural language task: distinguishing between human-written and LLM-generated English text using motif-based learning.

The repository includes:

-The original model code and our additions (Greedy Selector, Genetic Algorithm)

-Input sequences (data/)

-A new dataset (our_data/) and Python scripts for training set optimization 

-Reactivity analysis tools to evaluate classifier performance

All analyses from the original manuscript can be reproduced using the contents of model/ and data/, while new experiments and applications are documented below.

Steps 1–5 reproduce the baseline negative selection simulation described in the original paper. The steps after that extend the method to a new domain: distinguishing human vs LLM-generated natural language.


## About the model
The used model is a string-based model of thymic selection in the immune system.
It implements the techniques described in:

Johannes Textor, Katharina Dannenberg, Maciej Liskiewicz:
__A Generic Finite Automata Based Approach to Implementing Lymphocyte Repertoire Models.__
In _Proceedings of the 2014 conference on genetic and evolutionary computation (GECCO'14)_, pp. 129-137. ACM, 2014. http://dx.doi.org/10.1145/2576768.2598331

The code generates deterministic automata (DFAs) that recognize certain sets of strings of fixed length. The alphabet 
is defined in "proteins.hpp" and it is normally taken to be the 20-letter amino acid alphabet (of course, this can be 
adapted, and we also use "latinalphabet.hpp" in te current manuscript).

For example, the file "contiguous-fa.cpp" implements the so-called r-contiguous matching rule. 

## Dependencies and installation

You will need a C++ compiler and the OpenFST library binaries installed (http://www.openfst.org). On Mac OS X, you can 
install these with homebrew using

```
brew install openfst
```

OpenFST is also part of the libfst-dev package that you can install using the APT package
manager on Linux systems:

```
sudo apt-get install libfst-dev
```

To use the model code, first go to the `model/` folder and compile the code by typing

```
make
```

(If you do not have Make, you can also compile the code manually using the flags as shown in the `Makefile`).

This should work, but if you run into problems because the compiler cannot find the
OpenFST installation, consider setting the `-L` and `-I` flags in the `FSTFLAGS` variable
in the Makefile. 



## How to run the code


Using this code, we can now run a negative selection simulation in several steps.

### Step 1 : Generate an "automaton" description of a TCR repertoire containing all possible TCRs

We first generate a TCR repertoire that contains TCRs for all possible sequences. For the languages,
this is all possible 6-mer strings that can be generated using the letters a-z and the underscore
(which we use to replace any kind of interpunction). For the peptides, this is all possible 6-mer
peptides that can be generated from the 20 amino acids.

For the language repertoire, we run:

```
makerep-contiguous-fa-lang 6 3 | fstcompile --acceptor > complete-repertoire.fst
```

Here, the '6' specifies that we look at sequences of 6 letters/amino acids. The '3' is
the value of the t parameter. The process works the same for a TCR repertoire recognizing peptides,
except that we then use `makerep-contiguous-fa` instead of `makerep-contiguous-fa-lang`.



### Step 2 : Select and compress the training set

Next, we sample our training peptides. For example, to reproduce Figure 2C of the paper, we make
an empty training set for n = 0, or a training set of 500 english strings for n = 500. Example
files are included in the `example/` folder.

We then compress these for use by our program:

```
cat ../example/trainset-n0.txt | contiguous-fa-lang 6 3 | fstcompile --acceptor > trainset-n0-compressed.fst
cat ../example/trainset-n500.txt | contiguous-fa-lang 6 3 | fstcompile --acceptor > trainset-n500-compressed.fst
```

This produces a 'postively selected repertoire', so a compressed description of all TCRs recognizing one of the
sequences in the training set.

The process works the same for a TCR repertoire recognizing peptides,
except that we then use `contiguous-fa` instead of `contiguous-fa-lang`.


### Step 3 : Negatively select the repertoire

We now use the complete repertoire and the positively selected repertoire to arrive at a negatively selected repertoire.
In essence, this means that we remove all positively selected TCRs (which all recognize one of the TCRs in the trainset)
from the complete repertoire:

```
fstdifference complete-repertoire.fst trainset-n0-compressed.fst | fstminimize > repertoire-n0.fst
fstdifference complete-repertoire.fst trainset-n500-compressed.fst | fstminimize > repertoire-n500.fst
```
The process works exactly the same for a TCR repertoire recognizing peptides instead of languages.


### Step 4 : Count remaining TCRs in the selected repertoire

We can also count the number of TCRs left in the repertoire, which shows that there are fewer after selection on a non-empty training set:

```
cat repertoire-n0.fst | fstprint | countpaths
# yields 387420489

cat repertoire-n500.fst | fstprint | countpaths
# yields 359727693

```

### Step 5 : Count reacting TCRs for a test set

We can compute the number of reacting TCRs in the post-selection repertoire for a test set of "unseen" sequences that were *not*
part of the training set used for negative selection in step 2.

Examples of such test sets for Figure 2C are included in the `example/` folder.

We run:

```
contiguous-negative-selection-lang ../example/testset-english-n0.txt 6 3 < repertoire-n0.fst > test-english-n0.txt
contiguous-negative-selection-lang ../example/testset-english-n500.txt 6 3 < repertoire-n500.fst > test-english-n500.txt
contiguous-negative-selection-lang ../example/testset-xhosa-n0.txt 6 3 < repertoire-n0.fst > test-xhosa-n0.txt
contiguous-negative-selection-lang ../example/testset-xhosa-n500.txt 6 3 < repertoire-n500.fst > test-xhosa-n500.txt
```

Dividing these numbers by the total repertoire size obtained in step 4 and multiplying by 1 million yields Figure 2C.


### Extension – Distinguishing Human vs LLM-Generated Language

### Step 6 : Preprocess human and LLM text

```
python3 segment_text.py our_data/human_raw.txt our_data/human_clean.txt 6
python3 segment_text.py our_data/llm_raw.txt our_data/llm_clean.txt 6
```

### Step 7 : Generate a complete repertoire

```
makerep-contiguous-fa-lang 6 3 | fstcompile --acceptor > full_repertoire.fst
```

### Step 8 :  Train using a random sample of LLM-generated text

```
shuf our_data/llm_clean.txt | head -n 500 > our_data/train_llm_random.txt
cat our_data/train_llm_random.txt | ./contiguous-fa-lang 6 3 | fstcompile --acceptor > train_random.fst
fstdifference full_repertoire.fst train_random.fst | fstminimize > trained_repertoire.fst
```

### Step 9 :  Train using greedy motif coverage

```
python3 greedy_selector.py our_data/llm_clean.txt our_data/train_llm_greedy.txt 6 3 500
cat our_data/train_llm_greedy.txt | ./contiguous-fa-lang 6 3 | fstcompile --acceptor > train_greedy.fst
fstdifference full_repertoire.fst train_greedy.fst | fstminimize > trained_repertoire.fst

```

### Step 10 :   Train using genetic algorithm

```
python3 genetic_training.py
```

### Step 11 :   Train using genetic algorithm

```
python3 reactivity_analysis.py
```
