import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import tree, svm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlr
random_seed = 17

output_dir = r'C:\Users\Home\Desktop\ML\Project2\Results'

ds = ds.make_classification(n_samples=10000, n_features=10, n_informative=5, n_repeated=2,
                            n_clusters_per_class=5, flip_y=0.025, class_sep = 0.5, random_state=random_seed)

X, y = ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

grid_search_parameters = {
    'max_iters': [1000],                          # nn params
    'learning_rate': [0.1],                       # nn params
    'activation': [mlr.relu],            # nn params
    'restarts': [25],                             # rhc params
}

### RHC ###

nnr = mlr.NNGSRunner(
    x_train=X_train,
    y_train=y_train_hot,
    x_test=X_test,
    y_test=y_test_hot,
    experiment_name='nn_test_rhc',
    algorithm=mlr.algorithms.rhc.random_hill_climb,
    grid_search_parameters=grid_search_parameters,
    iteration_list=[1, 10, 50, 100, 250, 500, 1000],
    hidden_layer_sizes=[[40, 60]],
    bias=True,
    early_stopping=True,
    clip_max=5,
    max_attempts=500,
    n_jobs=5,
    seed=random_seed,
    output_directory=None
)

run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()

y_train_pred = grid_search_cv.predict(X_train)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print("")
print("RHC")
print(y_train_accuracy)
print(cv_results_df.T)

### SA ###

grid_search_parameters = {
    'max_iters': [1000],                          # nn params
    'learning_rate': [0.1],                       # nn params
    'activation': [mlr.relu],            # nn params
}

nnr = mlr.NNGSRunner(
    x_train=X_train,
    y_train=y_train_hot,
    x_test=X_test,
    y_test=y_test_hot,
    experiment_name='nn_test_sa',
    algorithm=mlr.algorithms.sa.simulated_annealing,
    grid_search_parameters=grid_search_parameters,
    iteration_list=[1, 10, 50, 100, 250, 500, 1000],
    hidden_layer_sizes=[[40, 60]],
    bias=True,
    early_stopping=True,
    clip_max=5,
    max_attempts=500,
    n_jobs=5,
    seed=random_seed,
    output_directory=None
)

run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()

y_train_pred = grid_search_cv.predict(X_train)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print("")
print("SA")
print(y_train_accuracy)
print(cv_results_df.T)

### GA ###

grid_search_parameters = {
    'max_iters': [1000],                          # nn params
    'learning_rate': [0.1],                       # nn params
    'activation': [mlr.relu],            # nn params
    'pop_size': [50],                           # rhc params
    'mutations': [0.05],
}

nnr = mlr.NNGSRunner(
    x_train=X_train,
    y_train=y_train_hot,
    x_test=X_test,
    y_test=y_test_hot,
    experiment_name='nn_test_ga',
    algorithm=mlr.algorithms.ga.genetic_alg,
    grid_search_parameters=grid_search_parameters,
    iteration_list=[1, 10, 50, 100, 250, 500, 1000],
    hidden_layer_sizes=[[40, 60]],
    bias=True,
    early_stopping=True,
    clip_max=5,
    max_attempts=500,
    n_jobs=5,
    seed=random_seed,
    output_directory=None
)

run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()

y_train_pred = grid_search_cv.predict(X_train)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print("")
print("SA")
print(y_train_accuracy)
print(cv_results_df.T)