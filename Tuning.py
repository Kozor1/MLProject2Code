import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr
import time
random_seed = 17

x_ax = []
fitness = []

# Vary the range of the tuning parameter for a valid number for the specific parameter to be tuned
for tun_param in np.linspace(0.01, 0.91, 9):

    x_ax.append(tun_param)

    # problem = mlr.generators.queens_generator.QueensGenerator.generate(size=100, seed=random_seed)
    # problem = mlr.generators.tsp_generator.TSPGenerator.generate(seed=random_seed, number_of_cities=50)
    # problem = mlr.generators.max_k_color_generator.MaxKColorGenerator.generate(seed=random_seed, number_of_nodes=100, max_colors=6)
    problem = mlr.generators.knapsack_generator.KnapsackGenerator.generate(seed=random_seed, number_of_items_types=100)


    # _, ga_best_fitness, _ = mlr.algorithms.genetic_alg(problem=problem, max_iters=2048, pop_size=50, mutation_prob=tun_param, random_state=random_seed)
    # fitness.append(ga_best_fitness)

    # _, rhc_best_fitness, _ = mlr.algorithms.random_hill_climb(problem=problem, max_iters=512, restarts=tun_param, random_state=random_seed)
    # fitness.append(rhc_best_fitness)

    # _, sa_best_fitness, _ = mlr.algorithms.simulated_annealing(problem=problem, max_iters=8192, schedule=mlr.algorithms.decay.GeomDecay(init_temp=tun_param), random_state=random_seed)
    # fitness.append(sa_best_fitness)

    _, mi_best_fitness, _ = mlr.algorithms.mimic(problem=problem, max_iters=512, pop_size=500, keep_pct=tun_param, random_state=random_seed)
    fitness.append(mi_best_fitness)

plt.plot(x_ax, fitness)

plt.title("Initial Temperature Tuning Curve")
plt.show()

