import random
import subprocess
import os
import matplotlib.pyplot as plt

# === CONFIGURATION ===
LLM_FILE = "our_data/test_llm_clean.txt"
HUMAN_FILE = "our_data/train_human_clean.txt"
WORK_DIR = "ga_output"
#POP_SIZE = 10
POP_SIZE = 20
CHROMOSOME_SIZE = 500
#GENERATIONS = 10
GENERATIONS = 20
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
    
    # Save training data relative to project root
    rel_train_path = f"{WORK_DIR}/train_{gen_id}_{ind_id}.txt"
    abs_train_path = os.path.abspath(rel_train_path)
    write_lines(rel_train_path, train_lines)

    # Inside model, refer to the file as "../ga_output/train_*.txt"
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
    score = sum(llm_scores) - sum(human_scores)
    return score


# === GA FUNCTIONS ===
def initialize_population(pool, pop_size, chromo_size):
    return [random.sample(pool, chromo_size) for _ in range(pop_size)]

def crossover(parent1, parent2):
    cut = len(parent1) // 2
    return parent1[:cut] + parent2[cut:], parent2[:cut] + parent1[cut:]

#def mutate(individual, pool, rate=0.05):
def mutate(individual, pool, rate=0.1):
    return [random.choice(pool) if random.random() < rate else x for x in individual]

# === MAIN GA LOOP ===
random.seed(SEED)
candidate_pool = load_lines(LLM_FILE)
print(f"Loaded {len(candidate_pool)} LLM lines")

population = initialize_population(candidate_pool, POP_SIZE, CHROMOSOME_SIZE)
fitness_log = []

for gen in range(GENERATIONS):
    print(f"\n=== Generation {gen} ===")
    fitnesses = []
    for i, ind in enumerate(population):
        score = evaluate_fitness(ind, gen, i)
        fitnesses.append((score, ind))
        fitness_log.append((gen, i, score))
        print(f"Individual {i} fitness: {score}")

    fitnesses.sort(reverse=True, key=lambda x: x[0])
    #top = [ind for (_, ind) in fitnesses[:2]]
    top = [ind for (_, ind) in fitnesses[:POP_SIZE // 2]]


    new_pop = top.copy()
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(top, 2)
        child1, child2 = crossover(p1, p2)
        new_pop.append(mutate(child1, candidate_pool))
        if len(new_pop) < POP_SIZE:
            new_pop.append(mutate(child2, candidate_pool))
    population = new_pop

# Save best result
best_score, best_set = fitnesses[0]
write_lines(f"{WORK_DIR}/best_training_set.txt", best_set)
print(f"\nBest score: {best_score} — saved to {WORK_DIR}/best_training_set.txt")

# Save fitness log
with open(f"{WORK_DIR}/fitness_log_2.csv", "w") as log_file:
    log_file.write("Generation,Individual,Fitness\n")
    for gen, ind, score in fitness_log:
        log_file.write(f"{gen},{ind},{score}\n")

# Plot fitness progression
plt.figure(figsize=(10,6))
for gen in range(GENERATIONS):
    scores = [score for g, i, score in fitness_log if g == gen]
    plt.plot([gen]*len(scores), scores, 'bo', alpha=0.6)
plt.title("Fitness over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.grid(True)
plt.savefig(f"{WORK_DIR}/fitness_plot_2.png")
print(f"\n Fitness plot saved to {WORK_DIR}/fitness_plot_2.png")
