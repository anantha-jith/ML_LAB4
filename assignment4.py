import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
n_samples, n_features = 100, 4
rng = np.random.RandomState(42)
X = rng.rand(n_samples, n_features)
y = rng.randint(0, 2, size=n_samples)
my_groups = rng.randint(0, 10, size=n_samples)
my_weights = rng.rand(n_samples)
my_other_weights = rng.rand(n_samples)
