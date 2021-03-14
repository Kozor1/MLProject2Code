import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr

random_seed = 17
output_dir = r'C:\Users\Home\Desktop\ML\Project2\Results'

problem = mlr.generators.max_k_color_generator.MaxKColorGenerator.generate(seed=random_seed, number_of_nodes=100, max_colors=6)
experiment_name = 'RHC_KColour'

rhc = mlr.RHCRunner(problem=problem,
                experiment_name=experiment_name,
                output_directory=output_dir,
                seed=random_seed,
                iteration_list=2 ** np.arange(10),
                max_attempts=5000,
                restart_list=[25, 75, 100])

# the two data frames will contain the results
df_run_stats, df_run_curves = rhc.run()
