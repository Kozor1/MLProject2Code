import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr

random_seed = 17
output_dir = r'C:\Users\Home\Desktop\ML\Project2\Results'

problem = mlr.generators.max_k_color_generator.MaxKColorGenerator.generate(seed=random_seed, number_of_nodes=100, max_colors=6)
experiment_name = 'GA_KColour'

ga = mlr.GARunner(problem=problem,
              experiment_name=experiment_name,
              output_directory=output_dir,
              seed=random_seed,
              iteration_list=2 ** np.arange(12),
              max_attempts=1000,
              population_sizes=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
              mutation_rates=np.linspace(0.01,0.5,50))

# the two data frames will contain the results
df_run_stats, df_run_curves = ga.run()
