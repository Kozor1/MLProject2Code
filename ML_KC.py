import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr

random_seed = 17
n = 100

output_dir = r'C:\Users\Home\Desktop\ML\Project2\Results'

problem = mlr.generators.max_k_color_generator.MaxKColorGenerator.generate(seed=random_seed, number_of_nodes=100, max_colors=6)
experiment_name = 'MI_KColour'

mmc = mlr.MIMICRunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=output_dir,
                  seed=random_seed,
                  iteration_list=2 ** np.arange(10),
                  max_attempts=500,
                  keep_percent_list=[0.25, 0.5, 0.75],
                      population_sizes= [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                      fevals=True,
                      use_fast_mimic=True)

df_run_stats, df_run_curves = mmc.run()