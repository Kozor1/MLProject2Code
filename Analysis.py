import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr

### TSP ###

# # SA
# sa_tsp = pd.read_csv('Results/SA_TSP/sa__SA_TSP__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/SA_TSP/sa__SA_TSP__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run)


# # RHC
# sa_tsp = pd.read_csv('Results/RHC_TSP/rhc__RHC_TSP__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/RHC_TSP/rhc__RHC_TSP__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)

# # GA
# sa_tsp = pd.read_csv('Results/GA_TSP/ga__GA_TSP__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/GA_TSP/ga__GA_TSP__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)

# MI
sa_tsp = pd.read_csv('Results/MI_TSP/mimic__MI_TSP__curves_df.csv', index_col=0)
sa_tsp_run = pd.read_csv('Results/MI_TSP/mimic__MI_TSP__run_stats_df.csv', index_col=0)

best_fitness = sa_tsp['Fitness'].min()
best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]

minimum_evaluations = best_runs['FEvals'].min()
best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
print(best_curve_run.T)



### K Colour ###

# # SA
# sa_tsp = pd.read_csv('Results/SA_KColour/sa__SA_KColour__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/SA_KColour/sa__SA_KColour__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].max()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run)

# # GA
# sa_tsp = pd.read_csv('Results/GA_KColour/ga__GA_KColour__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/GA_KColour/ga__GA_KColour__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)

# # RHC
# sa_tsp = pd.read_csv('Results/RHC_KColour/rhc__RHC_KColour__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/RHC_KColour/rhc__RHC_KColour__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)


# # MI
# sa_tsp = pd.read_csv('Results/MI_KColour/mimic__MI_KColour__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/MI_KColour/mimic__MI_KColour__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)

## N Queens ###

# # SA
# sa_tsp = pd.read_csv('Results/SA_NQueens/sa__SA_NQueens__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/SA_NQueens/sa__SA_NQueens__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run)

# # GA
# sa_tsp = pd.read_csv('Results/GA_NQueens/ga__GA_NQueens__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/GA_NQueens/ga__GA_NQueens__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)

# # RHC
# sa_tsp = pd.read_csv('Results/RHC_NQueens/rhc__RHC_NQueens__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/RHC_NQueens/rhc__RHC_NQueens__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)


# # MI
# sa_tsp = pd.read_csv('Results/MI_NQueens/mimic__MI_NQueens__curves_df.csv', index_col=0)
# sa_tsp_run = pd.read_csv('Results/MI_NQueens/mimic__MI_NQueens__run_stats_df.csv', index_col=0)
#
# best_fitness = sa_tsp['Fitness'].min()
# best_runs = sa_tsp[sa_tsp['Fitness'] == best_fitness]
#
# minimum_evaluations = best_runs['FEvals'].min()
# best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
# print(best_curve_run.T)