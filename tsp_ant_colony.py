"""
A.5 - Travelling Salesman Problem using Ant Colony Optimization
1. Ant System (AS) Algorithm
2. Max-Min Ant System (MMAS) Algorithm
3. Comparison of both algorithms
"""

import numpy as np
import time
import random

# ========================= GIVEN DATA =========================
# Distance matrix (5 cities)
d = np.array([
    [0, 10, 12, 11, 14],
    [10,  0, 13, 15,  8],
    [12, 13,  0,  9, 14],
    [11, 15,  9,  0, 16],
    [14,  8, 14, 16,  0]
], dtype=float)

# Initial pheromone matrix (all ones)
tho_init = np.ones((5, 5), dtype=float)

n_cities = 5

# ========================= COMMON PARAMETERS =========================
alpha = 1.0        # Pheromone importance factor
beta = 2.0         # Heuristic (visibility) importance factor
rho = 0.5          # Evaporation rate
n_ants = 5         # Number of ants
n_iterations = 100 # Number of iterations
Q = 100.0          # Pheromone deposit constant

# Heuristic information (visibility) = 1 / distance
eta = np.zeros_like(d)
for i in range(n_cities):
    for j in range(n_cities):
        if d[i][j] != 0:
            eta[i][j] = 1.0 / d[i][j]

# ========================= HELPER FUNCTIONS =========================

def calculate_tour_length(tour, dist_matrix):
    """Calculate total distance of a tour."""
    length = 0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i]][tour[i + 1]]
    length += dist_matrix[tour[-1]][tour[0]]  # Return to start
    return length


def select_next_city(pheromone, heuristic, current, visited, alpha, beta):
    """Select next city using roulette wheel selection based on transition probabilities."""
    n = len(pheromone)
    probabilities = np.zeros(n)

    for j in range(n):
        if j not in visited:
            probabilities[j] = (pheromone[current][j] ** alpha) * (heuristic[current][j] ** beta)

    total = probabilities.sum()
    if total == 0:
        # If all probabilities are zero, choose randomly from unvisited
        unvisited = [j for j in range(n) if j not in visited]
        return random.choice(unvisited)

    probabilities /= total

    # Roulette wheel selection
    r = random.random()
    cumulative = 0
    for j in range(n):
        cumulative += probabilities[j]
        if r <= cumulative:
            return j

    # Fallback
    unvisited = [j for j in range(n) if j not in visited]
    return unvisited[-1]


def construct_solution(pheromone, heuristic, n_cities, alpha, beta):
    """Construct a tour for one ant."""
    start = random.randint(0, n_cities - 1)
    tour = [start]
    visited = {start}

    for _ in range(n_cities - 1):
        next_city = select_next_city(pheromone, heuristic, tour[-1], visited, alpha, beta)
        tour.append(next_city)
        visited.add(next_city)

    return tour


# ======================================================================
#                      1. ANT SYSTEM (AS) ALGORITHM
# ======================================================================

def ant_system(dist_matrix, pheromone_init, eta, n_ants, n_iterations,
               alpha, beta, rho, Q):
    """
    Standard Ant System Algorithm.
    Pheromone update: All ants deposit pheromone on their paths.
    """
    n = len(dist_matrix)
    pheromone = pheromone_init.copy()

    best_tour = None
    best_length = float('inf')

    iteration_best_lengths = []

    for iteration in range(n_iterations):
        all_tours = []
        all_lengths = []

        # --- Step 1: Construct solutions for all ants ---
        for ant in range(n_ants):
            tour = construct_solution(pheromone, eta, n, alpha, beta)
            length = calculate_tour_length(tour, dist_matrix)
            all_tours.append(tour)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_tour = tour[:]

        iteration_best_lengths.append(min(all_lengths))

        # --- Step 2: Pheromone evaporation ---
        pheromone *= (1 - rho)

        # --- Step 3: Pheromone deposit (ALL ants contribute) ---
        for ant in range(n_ants):
            tour = all_tours[ant]
            length = all_lengths[ant]
            delta = Q / length

            for i in range(n - 1):
                pheromone[tour[i]][tour[i + 1]] += delta
                pheromone[tour[i + 1]][tour[i]] += delta  # Symmetric
            # Close the tour
            pheromone[tour[-1]][tour[0]] += delta
            pheromone[tour[0]][tour[-1]] += delta

    return best_tour, best_length, pheromone, iteration_best_lengths


# ======================================================================
#                  2. MAX-MIN ANT SYSTEM (MMAS) ALGORITHM
# ======================================================================

def max_min_ant_system(dist_matrix, pheromone_init, eta, n_ants, n_iterations,
                       alpha, beta, rho, Q, tau_max=5.0, tau_min=0.1):
    """
    Max-Min Ant System (MMAS) Algorithm.
    Key differences from standard AS:
      - Only the BEST ant (iteration-best or global-best) deposits pheromone.
      - Pheromone values are clamped within [tau_min, tau_max].
      - Pheromone trails are initialized to tau_max.
    """
    n = len(dist_matrix)
    pheromone = np.full((n, n), tau_max)  # Initialize to tau_max

    best_tour = None
    best_length = float('inf')

    iteration_best_lengths = []

    for iteration in range(n_iterations):
        all_tours = []
        all_lengths = []

        # --- Step 1: Construct solutions for all ants ---
        for ant in range(n_ants):
            tour = construct_solution(pheromone, eta, n, alpha, beta)
            length = calculate_tour_length(tour, dist_matrix)
            all_tours.append(tour)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_tour = tour[:]

        iteration_best_lengths.append(min(all_lengths))

        # Find the iteration-best ant
        iter_best_idx = np.argmin(all_lengths)
        iter_best_tour = all_tours[iter_best_idx]
        iter_best_length = all_lengths[iter_best_idx]

        # --- Step 2: Pheromone evaporation ---
        pheromone *= (1 - rho)

        # --- Step 3: Pheromone deposit (ONLY best ant deposits) ---
        # Use iteration-best for early iterations, global-best for later
        if iteration < n_iterations // 2:
            deposit_tour = iter_best_tour
            deposit_length = iter_best_length
        else:
            deposit_tour = best_tour
            deposit_length = best_length

        delta = Q / deposit_length

        for i in range(n - 1):
            pheromone[deposit_tour[i]][deposit_tour[i + 1]] += delta
            pheromone[deposit_tour[i + 1]][deposit_tour[i]] += delta
        pheromone[deposit_tour[-1]][deposit_tour[0]] += delta
        pheromone[deposit_tour[0]][deposit_tour[-1]] += delta

        # --- Step 4: Clamp pheromone within [tau_min, tau_max] ---
        pheromone = np.clip(pheromone, tau_min, tau_max)

    return best_tour, best_length, pheromone, iteration_best_lengths


# ======================================================================
#                          3. RUN AND COMPARE
# ======================================================================

def format_tour(tour):
    """Format tour as 1-indexed city names."""
    return " -> ".join([str(c + 1) for c in tour]) + " -> " + str(tour[0] + 1)


def run_comparison():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  TRAVELLING SALESMAN PROBLEM - ANT COLONY OPTIMIZATION")
    print("=" * 70)

    print("\n--- Given Distance Matrix ---")
    print(d.astype(int))

    print("\n--- Initial Pheromone Matrix ---")
    print(tho_init.astype(int))

    print(f"\n--- Parameters ---")
    print(f"  Alpha (pheromone weight)  : {alpha}")
    print(f"  Beta  (heuristic weight)  : {beta}")
    print(f"  Rho   (evaporation rate)  : {rho}")
    print(f"  Q     (deposit constant)  : {Q}")
    print(f"  Number of ants            : {n_ants}")
    print(f"  Number of iterations      : {n_iterations}")

    # -------------------- ANT SYSTEM --------------------
    print("\n" + "=" * 70)
    print("  1. ANT SYSTEM (AS) ALGORITHM")
    print("=" * 70)

    random.seed(42)
    np.random.seed(42)
    start_time = time.time()
    as_tour, as_length, as_pheromone, as_iter_best = ant_system(
        d, tho_init, eta, n_ants, n_iterations, alpha, beta, rho, Q
    )
    as_time = time.time() - start_time

    print(f"\n  Best Tour Found     : {format_tour(as_tour)}")
    print(f"  Best Tour Length    : {as_length}")
    print(f"  Execution Time      : {as_time:.6f} seconds")
    print(f"\n  Final Pheromone Matrix (AS):")
    print(np.round(as_pheromone, 3))

    # -------------------- MAX-MIN ANT SYSTEM --------------------
    print("\n" + "=" * 70)
    print("  2. MAX-MIN ANT SYSTEM (MMAS) ALGORITHM")
    print("=" * 70)

    random.seed(42)
    np.random.seed(42)
    start_time = time.time()
    mmas_tour, mmas_length, mmas_pheromone, mmas_iter_best = max_min_ant_system(
        d, tho_init, eta, n_ants, n_iterations, alpha, beta, rho, Q,
        tau_max=5.0, tau_min=0.1
    )
    mmas_time = time.time() - start_time

    print(f"\n  Best Tour Found     : {format_tour(mmas_tour)}")
    print(f"  Best Tour Length    : {mmas_length}")
    print(f"  Execution Time      : {mmas_time:.6f} seconds")
    print(f"\n  Final Pheromone Matrix (MMAS):")
    print(np.round(mmas_pheromone, 3))

    # -------------------- COMPARISON --------------------
    print("\n" + "=" * 70)
    print("  3. COMPARISON OF AS vs MMAS")
    print("=" * 70)

    print(f"""
    +-------------------------+------------------+------------------+
    |       Criterion         |   Ant System     |   Max-Min AS     |
    +-------------------------+------------------+------------------+
    | Best Tour Length        | {as_length:<16} | {mmas_length:<16} |
    | Best Tour               | {format_tour(as_tour):<16} | {format_tour(mmas_tour):<16} |
    | Execution Time (sec)    | {as_time:<16.6f} | {mmas_time:<16.6f} |
    | Convergence (iter 10)   | {as_iter_best[9]:<16} | {mmas_iter_best[9]:<16} |
    | Convergence (iter 50)   | {as_iter_best[49]:<16} | {mmas_iter_best[49]:<16} |
    | Convergence (iter 100)  | {as_iter_best[99]:<16} | {mmas_iter_best[99]:<16} |
    +-------------------------+------------------+------------------+
    """)

    print("  ANALYSIS:")
    print("  -" * 35)
    print("""
  TIME COMPLEXITY:
    * AS  Algorithm : O(n_iterations x n_ants x n_cities^2)
      - All ants deposit pheromone => O(n_ants x n_cities) per iteration update
    * MMAS Algorithm: O(n_iterations x n_ants x n_cities^2)
      - Only best ant deposits pheromone => O(n_cities) per iteration update
      - Additional clipping step => O(n_cities^2) per iteration
      - Overall same asymptotic complexity, but slightly less pheromone
        update work per iteration.

  SPACE COMPLEXITY:
    * Both algorithms: O(n_cities^2) for pheromone and distance matrices.

  KEY DIFFERENCES:
    * AS allows ALL ants to update pheromone => can lead to stagnation 
      as suboptimal paths accumulate pheromone.
    * MMAS restricts pheromone updates to BEST ant only and enforces
      pheromone bounds [tau_min, tau_max]:
        => Avoids premature convergence (pheromone capped at tau_max)
        => Maintains exploration (pheromone floored at tau_min)
    * MMAS generally produces better or equal solutions for larger 
      problem instances due to controlled exploration.
    """)

    if mmas_length < as_length:
        print("  [OK] MMAS found a BETTER solution than AS.")
    elif mmas_length == as_length:
        print("  [OK] Both algorithms found the SAME optimal solution.")
    else:
        print("  [OK] AS found a BETTER solution than MMAS in this run.")

    print("\n" + "=" * 70)
    print("  OPTIMAL TOUR (Brute Force Verification)")
    print("=" * 70)

    # Brute force for small problem to verify
    from itertools import permutations
    best_bf_length = float('inf')
    best_bf_tour = None
    for perm in permutations(range(n_cities)):
        length = calculate_tour_length(list(perm), d)
        if length < best_bf_length:
            best_bf_length = length
            best_bf_tour = list(perm)

    print(f"\n  Optimal Tour        : {format_tour(best_bf_tour)}")
    print(f"  Optimal Tour Length : {best_bf_length}")
    print(f"  AS  found optimal?  : {'Yes' if as_length == best_bf_length else 'No (gap: ' + str(as_length - best_bf_length) + ')'}")
    print(f"  MMAS found optimal? : {'Yes' if mmas_length == best_bf_length else 'No (gap: ' + str(mmas_length - best_bf_length) + ')'}")
    print()


if __name__ == "__main__":
    run_comparison()
