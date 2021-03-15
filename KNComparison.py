import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr
import time
random_seed = 17

size = 51

x_ax = []
rhc_times, rhc_fitness, sa_times, sa_fitness, ga_times, ga_fitness, mi_times, mi_fitness = [], [], [], [], [], [], [], []

for n in range(5, size, 5):

    x_ax.append(n)

    problem = mlr.generators.knapsack_generator.KnapsackGenerator.generate(seed= random_seed, number_of_items_types=n)

    rhc_start_time = time.time()
    _, rhc_best_fitness, _ = mlr.algorithms.random_hill_climb(problem=problem, max_iters=512, restarts=20, random_state=random_seed)
    rhc_time = time.time() - rhc_start_time
    rhc_fitness.append(rhc_best_fitness)
    rhc_times.append(rhc_time)

    sa_start_time = time.time()
    _, sa_best_fitness, _ = mlr.algorithms.simulated_annealing(problem=problem, max_iters=8192, schedule=mlr.algorithms.decay.GeomDecay(init_temp=200.0), random_state=random_seed)
    sa_time = time.time() - sa_start_time
    sa_fitness.append(sa_best_fitness)
    sa_times.append(sa_time)

    ga_start_time = time.time()
    _, ga_best_fitness, _ = mlr.algorithms.genetic_alg(problem=problem, max_iters=2048, pop_size=500, mutation_prob=0.35, random_state=random_seed)
    ga_time = time.time() - ga_start_time
    ga_fitness.append(ga_best_fitness)
    ga_times.append(ga_time)

    mi_start_time = time.time()
    _, mi_best_fitness, _ = mlr.algorithms.mimic(problem=problem, max_iters=512, pop_size=500, keep_pct=0.50, random_state=random_seed)
    mi_time = time.time() - mi_start_time
    mi_fitness.append(mi_best_fitness)
    mi_times.append(mi_time)

plt.plot(x_ax, rhc_fitness, label = 'rhc')
plt.plot(x_ax, sa_fitness, label = 'sa')
plt.plot(x_ax, ga_fitness, label = 'ga')
plt.plot(x_ax, mi_fitness, label = 'mi')

plt.title("Best fitness Curves")
plt.legend()
plt.show()

plt.plot(x_ax, rhc_times, label = 'rhc')
plt.plot(x_ax, sa_times, label = 'sa')
plt.plot(x_ax, ga_times, label = 'ga')
plt.plot(x_ax, mi_times, label = 'mi')

plt.title("Times")
plt.legend()
plt.show()
