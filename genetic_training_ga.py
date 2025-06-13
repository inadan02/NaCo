import random
import subprocess
import os
import matplotlib.pyplot as plt
import signal
import atexit

# === CONFIGURATION ===
LLM_FILE = "our_data/test_llm_clean.txt"
HUMAN_FILE = "our_data/train_human_clean.txt"
WORK_DIR = "ga_output"
RESULTS_ROOT = "my_ga_results"

#11450 = 10 * 10 
#1145 saniye 
POP_SIZE = 18
#POP_SIZE = 20
CHROMOSOME_SIZE = 500
#GENERATIONS = 10
GENERATIONS = 20
SEED = 42


# === SAFE EXIT ===
def cleanup_processes():
    try:
        subprocess.run("killall contiguous-negative-selection-lang", shell=True)
        print("[✓] Cleaned up lingering processes.")
    except:
        pass

atexit.register(cleanup_processes)


# === SAVE RUNS ===
def get_next_run_dir():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    existing = [d for d in os.listdir(RESULTS_ROOT) if d.startswith("run")]
    numbers = [int(d[3:]) for d in existing if d[3:].isdigit()]
    next_run = max(numbers, default=0) + 1
    run_dir = os.path.join(RESULTS_ROOT, f"run{next_run}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

RUN_DIR = get_next_run_dir()

def handle_sigterm(signum, frame):
    print("\n Received SIGTERM. Cleaning up...")
    cleanup_processes()
    exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGHUP, handle_sigterm)


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
            
def evaluate_fitness_parallel(args):
        ind, gen_id, ind_id = args
        score = evaluate_fitness(ind, gen_id, ind_id)
        return ind_id, score

def evaluate_fitness(train_lines, gen_id, ind_id):
    os.makedirs(WORK_DIR, exist_ok=True)
    
    rel_train_path = f"{WORK_DIR}/train_{gen_id}_{ind_id}.txt"
    write_lines(rel_train_path, train_lines)

    model_dir = "model"
    fst_train = f"train_{gen_id}_{ind_id}.fst"
    fst_full = f"full_{gen_id}_{ind_id}.fst"
    fst_rep = f"rep_{gen_id}_{ind_id}.fst"

    try:
        subprocess.run(
            f"cat ../{rel_train_path} | ./contiguous-fa-lang 6 3 | ~/openfst-1.6.3-install/bin/fstcompile --acceptor > {fst_train}",
            shell=True, cwd=model_dir, check=True
        )
        subprocess.run(
            f"./makerep-contiguous-fa-lang 6 3 | ~/openfst-1.6.3-install/bin/fstcompile --acceptor > {fst_full}",
            shell=True, cwd=model_dir, check=True
        )
        subprocess.run(
            f"~/openfst-1.6.3-install/bin/fstdifference {fst_full} {fst_train} | ~/openfst-1.6.3-install/bin/fstminimize > {fst_rep}",
            shell=True, cwd=model_dir, check=True
        )

        result_human = subprocess.run(
            f"./contiguous-negative-selection-lang ../{HUMAN_FILE} 6 3 < {fst_rep}",
            shell=True, cwd=model_dir, capture_output=True, text=True
        )
        result_llm = subprocess.run(
            f"./contiguous-negative-selection-lang ../{LLM_FILE} 6 3 < {fst_rep}",
            shell=True, cwd=model_dir, capture_output=True, text=True
        )

        human_scores = [int(x) for x in result_human.stdout.strip().split() if x.isdigit()]
        llm_scores   = [int(x) for x in result_llm.stdout.strip().split()   if x.isdigit()]
        # maximise foreign-minus-self
        score = sum(human_scores) - sum(llm_scores)

        human_sum = sum(human_scores)
        llm_sum = sum(llm_scores)
        total = human_sum + llm_sum

        # main objective
        gap = human_sum - llm_sum

        # -------- guard & slight penalty --------
        if total == 0:              # repertoire reacts to nothing
            score = -1e9            # make it unquestionably bad
        else:
            score = gap - 1e-4 * total   # prefer keeping some reactivity

        #print(f"G{gen_id} I{ind_id} | human={human_sum} llm={llm_sum} score={score}")



        

    except subprocess.CalledProcessError as e:
        print(f"[!] Error processing individual {ind_id} in generation {gen_id}: {e}")
        score = -1e9

    # Clean up FST files
    for filename in [fst_train, fst_full, fst_rep]:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    return score

# === GA FUNCTIONS ===
def initialize_population(pool, pop_size, chromo_size):
    return [random.sample(pool, chromo_size) for _ in range(pop_size)]

def crossover(parent1, parent2):
    cut = len(parent1) // 2
    return parent1[:cut] + parent2[cut:], parent2[:cut] + parent1[cut:]

def mutate(individual, pool, rate=0.2):
#def mutate(individual, pool, rate=0.1):
    return [random.choice(pool) if random.random() < rate else x for x in individual]

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    MAX_WORKERS = 2
    total_start = time.time()

    random.seed(SEED)
    candidate_pool = load_lines(LLM_FILE)
    print(f"Loaded {len(candidate_pool)} LLM lines")

    population = initialize_population(candidate_pool, POP_SIZE, CHROMOSOME_SIZE)
    fitness_log = []

    
    try:
        for gen in range(GENERATIONS):
            print(f"\n=== Generation {gen} ===")
            start_time = time.time()

            fitnesses = []
            args_list = [(ind, gen, i) for i, ind in enumerate(population)]

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(evaluate_fitness_parallel, args) for args in args_list]
                for future in as_completed(futures):
                    i, score = future.result()
                    fitnesses.append((score, population[i]))
                    fitness_log.append((gen, i, score))
                    print(f"Individual {i} fitness: {score}")

            print(f" Generation {gen} took {time.time() - start_time:.2f} seconds")

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

    except KeyboardInterrupt:
        print("\n Interrupted by user. Cleaning up...")
        cleanup_processes()
        exit(0)

    best_score, best_set = fitnesses[0]
    # Save best training set
    write_lines(os.path.join(RUN_DIR, "best_training_set.txt"), best_set)
    print(f"\nBest score: {best_score} — saved to {RUN_DIR}/best_training_set.txt")

    # Save fitness log
    fitness_log_path = os.path.join(RUN_DIR, "fitness_log.csv")
    with open(fitness_log_path, "w") as log_file:
        log_file.write("Generation,Individual,Fitness\n")
        for gen, ind, score in fitness_log:
            log_file.write(f"{gen},{ind},{score}\n")

    # Save fitness plot
    plt.figure(figsize=(10,6))
    for gen in range(GENERATIONS):
        scores = [score for g, i, score in fitness_log if g == gen]
        plt.plot([gen]*len(scores), scores, 'bo', alpha=0.6)
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True)
    fitness_plot_path = os.path.join(RUN_DIR, "fitness_plot.png")
    plt.savefig(fitness_plot_path)
    print(f"\nFitness plot saved to {fitness_plot_path}")

    # Save final rep.fst
    rep_path = os.path.join("model", f"rep_{GENERATIONS-1}_0.fst")
    final_rep_path = os.path.join(RUN_DIR, "rep.fst")
    if os.path.exists(rep_path):
        os.rename(rep_path, final_rep_path)
        print(f"\nSaved final rep.fst to {final_rep_path}")
    else:
        print("[!] rep.fst not found — might have been deleted.")

    # Save params
    with open(os.path.join(RUN_DIR, "params.txt"), "w") as f:
        f.write(f"POP_SIZE={POP_SIZE}\n")
        f.write(f"CHROMOSOME_SIZE={CHROMOSOME_SIZE}\n")
        f.write(f"GENERATIONS={GENERATIONS}\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"MAX_WORKERS={MAX_WORKERS}\n")
        f.write(f"MUTATION_RATE=0.1\n")
