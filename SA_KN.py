import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr

random_seed = 17
output_dir = r'C:\Users\Home\Desktop\ML\Project2\Results'

problem = mlr.generators.knapsack_generator.KnapsackGenerator.generate(seed= random_seed, number_of_items_types=100)
experiment_name = 'SA_KN'

sa = mlr.SARunner(problem=problem,
              experiment_name=experiment_name,
              output_directory=output_dir,
              seed=random_seed,
              iteration_list=2 ** np.arange(14),
              max_attempts=5000,
              temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
              decay_list=[mlr.GeomDecay])

# the two data frames will contain the results
df_run_stats, df_run_curves = sa.run()

