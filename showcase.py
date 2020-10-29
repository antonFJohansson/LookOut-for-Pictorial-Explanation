import numpy as np
from sklearn.datasets import load_iris
from LookOut import LookOut

data = load_iris()
X = data.data
axis_names = data.feature_names

num_plots = 3
num_outliers = 5

look = LookOut()
look.fit(X)
look.plot(num_plots, num_outliers, axis_names)
