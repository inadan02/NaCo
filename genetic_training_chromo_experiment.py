
import random
import subprocess
import os
import matplotlib.pyplot as plt

# === CONFIGURATION ===
LLM_FILE = "our_data/test_llm_clean.txt"
HUMAN_FILE = "our_data/train_human_clean.txt"
WORK_DIR = "ga_output"
POP_SIZE = 10
GENERATIONS = 10
SEED = 42

# === UTILS ===
def load_lines(file, max_lines=None):
    with open(file) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:max_lines] if max_lines else lines

def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

def evaluate_fitness(train_lines, gen_id, ind_id):
    os.makedirs(WORK_DIR, exist_ok=True)
    rel_train_path = f"{WORK_DIR}/train_{gen_id}_{ind_id}.txt"
    abs_train_path = os.path.abspath(rel_train_path)
    write_lines(rel_train_path, train_lines)
    subprocess.run(
        f"cat ../{rel_train_path} | ./contiguous-fa-lang 6 3 | fstcompile --acceptor > train.fst",
        shell=True, cwd="model"
    )
    subprocess.run(
        "./makerep-contiguous-fa-lang 6 3 | fstcompile --acceptor > full.fst",
        shell=True, cwd="model"
    )
    subprocess.run(
        "fstdifference full.fst train.fst | fstminimize > rep.fst",
        shell=True, cwd="model"
    )
    result_human = subprocess.run(
        f"./contiguous-negative-selection-lang ../{HUMAN_FILE} 6 3 < rep.fst",
        shell=True, cwd="model", capture_output=True, text=True
    )
    result_llm = subprocess.run(
        f"./contiguous-negative-selection-lang ../{LLM_FILE} 6 3 < rep.fst",
        shell=True, cwd="model", capture_output=True, text=True
    )
    human_scores = [int(x) for x in result_human.stdout.strip().split() if x.isdigit()]
    llm_scores = [int(x) for x in result_llm.stdout.strip().split() if x.isdigit()]
    return sum(llm_scores) - sum(human_scores)

# === GA FUNCTIONS ===
def initialize_population(pool, pop_size, chromo_size):
    return [random.sample(pool, chromo_size) for _ in range(pop_size)]

def crossover(parent1, parent2):
    cut = len(parent1) // 2
    return parent1[:cut] + parent2[cut:], parent2[:cut] + parent1[cut:]

def mutate(individual, pool, rate=0.1):
    return [random.choice(pool) if random.random() < rate else x for x in individual]

# === MAIN LOOP OVER CHROMOSOME SIZES ===
random.seed(SEED)
candidate_pool = load_lines(LLM_FILE)
CHROMOSOME_SIZES = [500, 600, 700]
fitness_by_size = []

for chromo_size in CHROMOSOME_SIZES:
    print(f"\n=== Chromosome Size: {chromo_size} ===")
    population = initialize_population(candidate_pool, POP_SIZE, chromo_size)
    fitness_log = []

    for gen in range(GENERATIONS):
        print(f"Generation {gen}")
        fitnesses = []
        for i, ind in enumerate(population):
            score = evaluate_fitness(ind, gen, i)
            fitnesses.append((score, ind))
            fitness_log.append((gen, i, score))
            print(f"Individual {i} fitness: {score}")

        fitnesses.sort(reverse=True, key=lambda x: x[0])
        top = [ind for (_, ind) in fitnesses[:POP_SIZE // 2]]
        new_pop = top.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(top, 2)
            child1, child2 = crossover(p1, p2)
            new_pop.append(mutate(child1, candidate_pool))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(child2, candidate_pool))
        population = new_pop

    best_score = fitnesses[0][0]
    fitness_by_size.append((chromo_size, best_score))
    print(f"Best score for size {chromo_size}: {best_score}")

# === SAVE CSV ===
csv_path = f"{WORK_DIR}/fitness_by_chromosome_size.csv"
with open(csv_path, "w") as f:
    f.write("Chromosome_Size,Best_Fitness\n")
    for size, score in fitness_by_size:
        f.write(f"{size},{score}\n")

# === PLOT FITNESS VS SIZE ===
sizes = [x[0] for x in fitness_by_size]
scores = [x[1] for x in fitness_by_size]

plt.figure(figsize=(8, 5))
plt.plot(sizes, scores, marker='o')
plt.title("Best Fitness vs Chromosome Size")
plt.xlabel("Chromosome Size")
plt.ylabel("Best Fitness (higher is better)")
plt.grid(True)
plt.savefig(f"{WORK_DIR}/fitness_vs_chromosome_size.png")
plt.show()
