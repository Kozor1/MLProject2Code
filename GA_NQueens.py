import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr

random_seed = 17
output_dir = r'C:\Users\Home\Desktop\ML\Project2\Results'

problem = mlr.generators.queens_generator.QueensGenerator.generate(seed=random_seed, size=50)
experiment_name = 'GA_NQueens'

ga = mlr.GARunner(problem=problem,
              experiment_name=experiment_name,
              output_directory=output_dir,
              seed=random_seed,
              iteration_list=2 ** np.arange(12),
              max_attempts=1000,
              population_sizes=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
              mutation_rates=np.linspace(0.01,0.41,10))

# the two data frames will contain the results
df_run_stats, df_run_curves = ga.run()
