import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import tree, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time
plt.rcParams["figure.figsize"] = (3.5,2.5)

random_seed = 17

ds = ds.make_classification(n_samples=10000, n_features=10, n_informative=5, n_repeated=2,
                            n_clusters_per_class=5, flip_y=0.025, class_sep = 0.5, random_state=random_seed)

X, y = ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

opt_clf = MLPClassifier(activation='logistic', hidden_layer_sizes=[40,60], solver='lbfgs', alpha=0.1, random_state=random_seed, max_iter=10000)

start = time.time()
model = opt_clf.fit(X_train, y_train)
fit_time = time.time() - start

y_preds = opt_clf.predict(X_test)

train_score = accuracy_score(opt_clf.predict(X_train), y_train)
test_score = accuracy_score(y_test, y_preds)

print(fit_time)
print(train_score)
print(test_score)

